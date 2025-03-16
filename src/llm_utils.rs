use anyhow::Context as AnyhowContext;
use encoding_rs::UTF_8;
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaModel, Special, params::LlamaModelParams},
    sampling::LlamaSampler,
};
use std::{num::NonZeroU32, pin::pin};
pub struct LlamaApp {
    backend: LlamaBackend,
    model: LlamaModel,
}

impl LlamaApp {
    /// Creates a new instance by loading a given model file from disk.
    pub fn new(model_path: &str) -> anyhow::Result<Self> {
        // Initialize the backend
        let mut backend = LlamaBackend::init().context("Failed to initialize LLaMA backend")?;
        backend.void_logs(); // => remove this line if you want to see the logs
        // Set up model parameters (you can customize as needed)
        let model_params = LlamaModelParams::default().with_use_mlock(true);
        let model_params = pin!(model_params);
        // Load the model from file
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .with_context(|| format!("Unable to load model from path: {model_path}"))?;
        Ok(Self { backend, model })
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
        // Create context parameters (controls how many tokens the context can hold)
        let ctx_params =
            LlamaContextParams::default().with_n_ctx(Some(NonZeroU32::new(ctx_size).unwrap()));
        // Create a context for this model
        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .context("Unable to create LLaMA context")?;

        // Build a sampler (decides how to pick tokens)
        let mut sampler = build_sampler(seed, temp);

        // Convert prompt to tokens (including a BOS token at the start)
        let tokens = self
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
        // We'll generate until we hit 1000 tokens or an EOG (end-of-generation) token
        while n_cur <= max_generation_tokens {
            // 1) Sample next token
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            // Accept the token (update internal state in the sampler, if any)
            sampler.accept(token);

            // 2) Check for end-of-generation token
            if self.model.is_eog_token(token) {
                // Stop generation
                break;
            }

            // 3) Convert token to UTF-8 and append to output
            let token_bytes = self.model.token_to_bytes(token, Special::Tokenize)?;
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

        Ok(output_text)
    }
}

fn build_sampler(seed: Option<u32>, temp: f32) -> LlamaSampler {
    LlamaSampler::chain_simple([
        LlamaSampler::dist(seed.unwrap_or_default()),
        LlamaSampler::top_k(40),
        LlamaSampler::top_p(0.95, 10),
        LlamaSampler::temp(temp.max(0.2)),
        LlamaSampler::mirostat_v2(seed.unwrap_or_default(), 5.0, 0.1),
        LlamaSampler::penalties(64, 1.05, 0.0, 0.0),
    ])
}

// LLM utility functions for datafusion UDFs
