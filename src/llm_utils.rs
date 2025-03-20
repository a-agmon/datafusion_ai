use anyhow::Context as AnyhowContext;
use encoding_rs::UTF_8;
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaModel, Special, params::LlamaModelParams},
    sampling::LlamaSampler,
};
use std::collections::VecDeque;
use std::{
    num::NonZeroU32,
    pin::pin,
    sync::{Arc, Mutex, RwLock},
};

struct LlamaResources {
    backend: LlamaBackend,
    model: LlamaModel,
}

impl std::fmt::Debug for LlamaResources {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaResources").finish_non_exhaustive()
    }
}

static LLAMA_RESOURCES: tokio::sync::OnceCell<Mutex<LlamaResources>> =
    tokio::sync::OnceCell::const_new();

#[derive(Debug)]
pub struct LlamaApp {}

impl LlamaApp {
    /// Creates a new instance by loading a given model file from disk.
    pub fn new(model_path: &str) -> anyhow::Result<Self> {
        // Initialize the backend
        let mut backend = LlamaBackend::init().context("Failed to initialize LLaMA backend")?;
        backend.void_logs(); // => remove this line if you want to see the logs

        let model_params = LlamaModelParams::default();
        let model_params = pin!(model_params);
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .with_context(|| format!("Unable to load model from path: {}", model_path))?;

        let resources = LlamaResources { backend, model };
        LLAMA_RESOURCES.set(Mutex::new(resources)).unwrap();

        Ok(Self {})
    }

    /// Generates text given a prompt.
    /// This reuses the model + context stored in `self`.
    pub fn generate_text(
        &self,
        prompt: &str,
        ctx_size: u32,
        temp: f32,
        seed: Option<u32>,
    ) -> anyhow::Result<String> {
        let ctx_params =
            LlamaContextParams::default().with_n_ctx(Some(NonZeroU32::new(ctx_size).unwrap()));

        let resources = LLAMA_RESOURCES.get().unwrap().lock().unwrap();

        let output_text = {
            // Create a context for this model
            let mut ctx = resources
                .model
                .new_context(&resources.backend, ctx_params)
                .context("Unable to create LLaMA context")?;

            // Build a sampler (decides how to pick tokens)
            let mut sampler = build_sampler(seed, temp);

            // Convert prompt to tokens (including a BOS token at the start)
            let tokens = resources
                .model
                .str_to_token(prompt, AddBos::Always)
                .with_context(|| format!("Failed to tokenize prompt: {prompt}"))?;

            let prompt_length = tokens.len() as i32;

            // Prepare batch
            let batch_size = std::cmp::max(prompt_length, 64) as usize;
            let mut batch = LlamaBatch::new(batch_size, 1);
            let last_index = prompt_length - 1;
            for (i, token) in (0_i32..).zip(tokens.into_iter()) {
                let is_last = i == last_index;
                batch.add(token, i, &[0], is_last)?;
            }

            // Decode the prompt (feed prompt tokens into context)
            ctx.decode(&mut batch)?;

            // Main generation loop: repeatedly sample the next token
            let mut output_text = String::new();
            let max_generation_tokens = (ctx_size as i32) - prompt_length;
            let mut n_cur = batch.n_tokens();
            // We'll generate until we hit max tokens or an EOG (end-of-generation) token
            while n_cur <= max_generation_tokens {
                // 1) Sample next token
                let token = sampler.sample(&ctx, batch.n_tokens() - 1);
                // Accept the token (update internal state in the sampler, if any)
                sampler.accept(token);

                // 2) Check for end-of-generation token
                if resources.model.is_eog_token(token) {
                    // Stop generation
                    break;
                }

                // 3) Convert token to UTF-8 and append to output
                let token_bytes = resources.model.token_to_bytes(token, Special::Tokenize)?;
                let mut decode_buffer = String::with_capacity(32);
                {
                    let mut decoder = UTF_8.new_decoder();
                    let _ = decoder.decode_to_string(&token_bytes, &mut decode_buffer, false);
                }
                output_text.push_str(&decode_buffer);

                // 4) Feed the newly generated token back into the model so it can predict the next one
                batch.clear();
                batch.add(token, n_cur, &[0], true)?;
                ctx.decode(&mut batch)?;

                n_cur += 1;
            }
            output_text
        }; // context is dropped here

        Ok(output_text)
    }
}

/// Build the sampler (decides how to pick next tokens).
fn build_sampler(seed: Option<u32>, temp: f32) -> LlamaSampler {
    // A sampler pipeline: random distribution + greedy pick.
    // You can extend or replace with your own logic (top-k, top-p, etc.)
    let sampler = LlamaSampler::chain_simple([
        LlamaSampler::dist(seed.unwrap_or(1234)),
        LlamaSampler::greedy(),
        LlamaSampler::temp(temp),
        //LlamaSampler::min_p(0.2, 10),
    ]);
    sampler
}

/// Helper function to create a prompt for the LLM
pub fn get_prompt(instruction: &str, column_values: &[String]) -> String {
    let column_values_str = column_values
        .iter()
        .enumerate()
        .map(|(i, value)| format!("{}. {}", i + 1, value))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an AI evaluator that processes lists of items according to specific criteria.
Always respond with ONLY comma-separated values matching the exact number and order of input items. 
For ratings, use only the specified numbers, or categories, For yes/no questions, use only 'yes' or 'no'.
<|eot_id|><|start_header_id|>user<|end_header_id|>
{instruction}:
{column_values_str}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_llama_app_creation() {
        // Replace with a path to a real model for testing
        let model_path = "models/llama_df_ai.Q4_K_M.gguf";
        let llama_app = LlamaApp::new(model_path).unwrap();
        let prompt = get_prompt(
            "Rate how likely these customers are to become repeat buyers as likely, neutral, or unlikely",
            &[
                "Excellent experience!".to_string(),
                "Wrong item delivered.".to_string(),
                "Fast delivery, great service!".to_string(),
            ],
        );
        println!("prompt: {}", prompt);
        let res = llama_app.generate_text(&prompt, 512, 0.1, None).unwrap();
        println!("res: {}", res);
        assert_eq!(res.trim(), "1 -> likely\n2 -> unlikely\n3 -> likely");
    }
}
