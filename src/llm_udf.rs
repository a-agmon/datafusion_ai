use datafusion::arrow::array::StringArray;
use datafusion::arrow::datatypes::DataType;
use datafusion_common::cast::as_string_array;
use datafusion_common::{DataFusionError, Result, ScalarValue, plan_err};
use datafusion_doc::Documentation;
use datafusion_expr::ScalarUDFImpl;
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, Signature, Volatility};
use datafusion_macros::user_doc;
use rayon::prelude::*;
use std::any::Any;
use std::sync::Arc;

use crate::llm_utils::{LlamaApp, get_prompt};

/// This struct for a simple UDF that adds one to an int32
#[user_doc(
    doc_section(label = "AI functions"),
    description = "Ask LLM",
    syntax_example = "ask_llm('instruction', 'column_value')"
)]
#[derive(Debug)]
pub struct AskLLM {
    signature: Signature,
    llama_app: LlamaApp,
}

impl AskLLM {
    /// Create a new instance of the UDF
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![DataType::Utf8, DataType::Utf8],
                Volatility::Immutable,
            ),
            llama_app: LlamaApp::new("models/llama_df_ai.Q4_K_M.gguf")
                .expect("Failed to initialize LlamaApp"),
        }
    }

    fn process_chunk(&self, instruction: &str, vals: &[String]) -> Result<Vec<String>> {
        let mut records_outcome: Vec<String> = Vec::with_capacity(vals.len());
        if vals.is_empty() {
            println!("vals is empty");
            return Ok(records_outcome);
        }
        let prompt = get_prompt(instruction, vals);
        let text = self
            .llama_app
            .generate_text(&prompt, 512, 0.0, None)
            .map_err(|e| DataFusionError::Internal(e.to_string()))?;
        let values: Vec<String> = parse_llm_response(&text);
        if values.len() == vals.len() {
            records_outcome.extend(values);
        } else {
            let message = format!(
                "Error: mismatched result count: {} != {}. results: {:?}",
                values.len(),
                vals.len(),
                values
            );
            records_outcome.extend(vec![message; vals.len()]);
        }

        Ok(records_outcome)
    }
}

/// Implement the ScalarUDFImpl trait for AskLLM
impl ScalarUDFImpl for AskLLM {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "ask_llm"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, args: &[DataType]) -> Result<DataType> {
        if !matches!(args.get(0), Some(&DataType::Utf8)) {
            return plan_err!("ask_llm only accepts Utf8 arguments");
        }
        Ok(DataType::Utf8)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let ScalarFunctionArgs { mut args, .. } = args;
        assert_eq!(args.len(), 2);
        let instruction = args.pop().unwrap();
        let column_value = args.pop().unwrap();
        assert_eq!(instruction.data_type(), DataType::Utf8);
        assert_eq!(column_value.data_type(), DataType::Utf8);

        match (instruction, column_value) {
            (
                ColumnarValue::Array(col_values),
                ColumnarValue::Scalar(ScalarValue::Utf8(instruction)),
            ) => {
                let col_values = as_string_array(col_values.as_ref())?;
                println!("instruction: {:?}", instruction);
                let values: Vec<_> = col_values.iter().collect();
                let chunk_size = 10;
                // Process chunks in parallel using Rayon
                let result: Vec<String> = values
                    //.par_chunks(chunk_size)
                    .chunks(chunk_size)
                    .flat_map(|chunk| {
                        let vals: Vec<String> = chunk
                            .iter()
                            .map(|opt| opt.unwrap_or_default().to_string())
                            .collect();
                        let instruction_str = instruction.as_deref().unwrap_or_default();
                        match self.process_chunk(instruction_str, &vals) {
                            Ok(records) => records,
                            Err(e) => vec![format!("Error processing chunk: {}", e); vals.len()],
                        }
                    })
                    .collect();

                Ok(ColumnarValue::Array(Arc::new(StringArray::from(result))))
            }

            _ => {
                return plan_err!(
                    "ask_llm only accepts 2 arguments in the form of 'instruction' (string), 'column_value' (column)"
                );
            }
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

fn parse_llm_response(input: &str) -> Vec<String> {
    input
        .lines()
        .filter_map(|line| {
            line.split("->")
                .nth(1)
                .map(|value| value.trim().to_string())
        })
        .collect()
}
