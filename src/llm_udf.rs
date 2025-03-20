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
use std::time::Instant;

use crate::ollama_utils::OllamaApp;

// Thread-local runtime creator function so that we can use async calls in sync contexts
fn create_tokio_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime")
}

#[user_doc(
    doc_section(label = "AI functions"),
    description = "Ask LLM",
    syntax_example = "ask_llm('instruction', 'column_value')"
)]
#[derive(Debug)]
pub struct AskLLM {
    signature: Signature,
    ollama_model: String,
    ollama_url: String,
}

impl AskLLM {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![DataType::Utf8, DataType::Utf8],
                Volatility::Immutable,
            ),
            ollama_model: "llama32-df:latest".to_string(),
            ollama_url: "http://localhost:11434/api/chat".to_string(),
        }
    }

    // this function process a chunk of rows and will be called in parallel using rayon
    async fn process_chunk(&self, instruction: &str, vals: &[String]) -> Result<Vec<String>> {
        let mut records_outcome: Vec<String> = Vec::with_capacity(vals.len());
        if vals.is_empty() {
            println!("vals is empty");
            return Ok(records_outcome);
        }
        let ollama_app = OllamaApp::new(&self.ollama_model, &self.ollama_url)
            .map_err(|e| DataFusionError::Internal(e.to_string()))?;

        let llm_response = ollama_app
            .generate_text(instruction, vals)
            .await
            .map_err(|e| DataFusionError::Internal(e.to_string()))?;

        let evaluated_values: Vec<String> = parse_llm_response(&llm_response);
        // sanity check that the number of results is the same as the number of input values
        if evaluated_values.len() == vals.len() {
            records_outcome.extend(evaluated_values);
        } else {
            let error_message = format!(
                "Error: mismatched result count: {} != {}. results: {:?}",
                evaluated_values.len(),
                vals.len(),
                evaluated_values
            );
            records_outcome.extend(vec![error_message; vals.len()]);
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
                let chunk_size = 5;

                let result: Vec<String> = values
                    .par_chunks(chunk_size)
                    .flat_map(|chunk| {
                        // first we extract the column values from the chunk
                        let vals: Vec<String> = chunk
                            .iter()
                            .map(|opt| opt.unwrap_or_default().to_string())
                            .collect();
                        // then we extract the instruction in the UDF
                        let instruction_str = instruction.as_deref().unwrap_or_default();
                        let time_start = Instant::now();
                        let rt = create_tokio_runtime();
                        println!("runtime created in {:?}", time_start.elapsed());
                        match rt.block_on(self.process_chunk(instruction_str, &vals)) {
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
