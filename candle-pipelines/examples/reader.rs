use candle_pipelines::error::Result;
use candle_pipelines::reader::ReaderPipelineBuilder;
use futures::StreamExt;
use std::fs;
use std::io::{self, Write};
use std::pin::pin;

#[tokio::main]
async fn main() -> Result<()> {
    // Fetch HTML from cursor.com
    println!("Fetching HTML from https://cursor.com/home ...");
    let html = reqwest::get("https://cursor.com/home")
        .await
        .expect("Failed to fetch URL")
        .text()
        .await
        .expect("Failed to read response body");

    println!("Fetched {} bytes of HTML\n", html.len());

    // Build the pipeline
    let pipeline = ReaderPipelineBuilder::new().cuda(0).build().await?;

    // Convert to markdown with streaming
    println!("Converting to Markdown (streaming)...\n");
    println!("---");

    let mut output = String::new();
    let stream = pipeline.to_markdown_stream(&html)?;
    let mut stream = pin!(stream);

    while let Some(chunk) = stream.next().await {
        let text = chunk?;
        print!("{}", text);
        io::stdout().flush().unwrap();
        output.push_str(&text);
    }

    println!("\n---\n");

    // Save to file
    fs::write("reader_example.md", &output).expect("Failed to write output file");
    println!("Saved to reader_example.md ({} chars)", output.len());

    Ok(())
}
