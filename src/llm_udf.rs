use datafusion::arrow::array::{ArrayRef, Int64Array, StringArray};
use datafusion::arrow::datatypes::DataType;
use datafusion_common::cast::{as_int64_array, as_string_array};
use datafusion_common::{DataFusionError, Result, ScalarValue, plan_err};
use datafusion_doc::Documentation;
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, Signature, Volatility, col};
use datafusion_expr::{ScalarUDF, ScalarUDFImpl};
use datafusion_macros::user_doc;
use std::any::Any;
use std::sync::Arc;
use std::sync::LazyLock;

/// This struct for a simple UDF that adds one to an int32
#[user_doc(
    doc_section(label = "AI functions"),
    description = "Ask LLM",
    syntax_example = "ask_llm('instruction', 'column_value')"
)]
#[derive(Debug)]
pub struct AskLLM {
    signature: Signature,
    aliases: Vec<String>,
}

impl AskLLM {
    /// Create a new instance of the `PowUdf` struct
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(
                // this function will always take two arguments of type string
                vec![DataType::Utf8, DataType::Utf8],
                // this function is deterministic and will always return the same
                // result for the same input
                Volatility::Immutable,
            ),
            // we will also add an alias of "my_pow"
            aliases: vec!["ask_llm".to_string()],
        }
    }
}

/// Implement the ScalarUDFImpl trait for AddOne
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
    // The actual implementation would add one to the argument
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
                let result = col_values
                    .iter()
                    .map(|value| {
                        // just count the number of words in the value
                        let num_words = value.unwrap_or_default().split_whitespace().count();
                        format!("{} words", num_words)
                    })
                    .collect::<Vec<String>>();
                Ok(ColumnarValue::Array(Arc::new(StringArray::from(result))))
            }

            _ => {
                return plan_err!("ask_llm only accepts Utf8 arguments");
            }
        }
    }
    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}
