use std::time::Instant;

use datafusion::prelude::*;
use datafusion_expr::ScalarUDF;
mod llm_udf;
mod llm_utils;
#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    // register the table
    let ctx = SessionContext::new();
    ctx.register_csv(
        "sample_table",
        "sample_data/sample_data.csv",
        CsvReadOptions::new(),
    )
    .await?;

    // create a plan to run a SQL query
    let df = ctx
        .sql("SELECT `Order ID`, `Customer ID`, `Customer Feedback` FROM sample_table limit 10")
        .await?;

    // execute and print results
    df.show().await?;

    let ask_llm = ScalarUDF::from(llm_udf::AskLLM::new());
    ctx.register_udf(ask_llm.clone());
    let query = r#"
    SELECT 
        "Order ID", "Customer ID", "Customer Feedback", 
        ask_llm('Rate the satisfaction level of the customer as satisfied, neutral, or dissatisfied', "Customer Feedback") 
        as satisfaction_level
    FROM sample_table 
    limit 50
    "#;
    let time_start = Instant::now();
    let df = ctx.sql(query).await?;
    df.show().await?;
    let time_end = Instant::now();
    println!("Time taken: {:?}", time_end.duration_since(time_start));
    Ok(())
}
