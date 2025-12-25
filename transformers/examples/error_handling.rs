use transformers::error::{Result, TransformersError};

#[tokio::main]
async fn main() -> Result<()> {
    // Create a fake error to demonstrate error handling
    let fake_error = TransformersError::Download(
        "Download timed out for 'made-up-file' from 'fake-repo' after 67 attempt(s)".to_string(),
    );

    let result = Result::<()>::Err(fake_error);

    // Match on error variants - each maps to a different user action
    match &result {
        Ok(_) => unreachable!(),
        Err(TransformersError::Download(_)) => {
            println!("Download error - retry with backoff");
            println!("{:?}", result);
        }
        Err(TransformersError::Tokenization(_)) => {
            println!("Tokenization error - check input text");
            println!("{:?}", result);
        }
        Err(TransformersError::Tool(_)) => {
            println!("Tool error - fix tool configuration");
            println!("{:?}", result);
        }
        Err(TransformersError::Device(_)) => {
            println!("Device error - fall back to CPU");
            println!("{:?}", result);
        }
        Err(TransformersError::Unexpected(_)) => {
            println!("Unexpected error - report bug");
            println!("{:?}", result);
        }
        // Required for #[non_exhaustive] enum - future variants
        Err(_) => {
            println!("Unknown error variant");
            println!("{:?}", result);
        }
    };

    Ok(())
}
