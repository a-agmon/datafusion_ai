use anyhow::Context as AnyhowContext;
use serde_json::{Value, json};


#[derive(Debug, Clone)]
pub struct OllamaApp {
    model_name: String,
    url: String,
    client: reqwest::Client,
}

impl OllamaApp {
    /// Creates a new instance for interacting with the Ollama server.
    pub fn new(model_name: &str, url: &str) -> anyhow::Result<Self> {
        Ok(Self {
            model_name: model_name.to_string(),
            url: url.to_string(),
            client: reqwest::Client::new(),
        })
    }

    /// Generates text by sending a prompt to the Ollama server.
    pub async fn generate_text(
        &self,
        instruction: &str,
        column_values: &[String],
    ) -> anyhow::Result<String> {
        // Format the content string
        let content = format_content(instruction, column_values);

        // Build the request JSON directly
        let request = json!({
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "stream": false
        });

        let response = self
            .client
            .post(&self.url)
            .json(&request)
            .send()
            .await
            .context("Failed to send request to Ollama server")?;

        // Parse the response
        let response_text = response.text().await?;

        let json: Value =
            serde_json::from_str(&response_text).context("Failed to parse Ollama response")?;

        // Extract the message content
        let content = json["message"]["content"]
            .as_str()
            .unwrap_or("No response content")
            .to_string();

        Ok(content)
    }
}

/// Helper function to format the content for the prompt
fn format_content(instruction: &str, column_values: &[String]) -> String {
    let column_values_str = column_values
        .iter()
        .enumerate()
        .map(|(i, value)| format!("{}. {}", i + 1, value))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"{instruction}:
{column_values_str}"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ollama_app() {
        let ollama_app =
            OllamaApp::new("llama32-df:latest", "http://localhost:11434/api/chat").unwrap();
        let values = vec![
            "Excellent experience!".to_string(),
            "Wrong item delivered.".to_string(),
            "The product was defective.".to_string(),
            "Fast delivery, great service!".to_string(),
        ];

        let res = ollama_app
            .generate_text(
                "Rate how likely these customers are to become repeat buyers as likely, neutral, or unlikely",
                &values
            )
            .await
            .unwrap();

        // Assert the exact expected format
        assert!(
            res.contains("1 -> likely")
                && res.contains("2 -> unlikely")
                && res.contains("3 -> unlikely")
                && res.contains("4 -> likely")
        );
    }
}
