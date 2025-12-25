use transformers::error::{DownloadError, Result, TransformersError};

#[tokio::main]
async fn main() -> Result<()> {
    let fake_error = TransformersError::Download(DownloadError::Timeout {
        repo: "fake-repo".to_string(),
        file: "made-up-file".to_string(),
        attempts: 67,
    });

    let result = Result::<TransformersError>::Err(fake_error);

    match &result {
        Ok(_) => unreachable!(),
        Err(TransformersError::Download(DownloadError::Timeout { .. })) => {
            println!("Download error, retrying...");
            println!("{:?}", result);
        }
        Err(TransformersError::Download(DownloadError::ApiInit { reason })) => {
            println!("API init error: {}", reason);
            println!("{:?}", result);
        }
        Err(TransformersError::Download(DownloadError::Failed { .. })) => {
            println!("Download failed, retrying...");
            println!("{:?}", result);
        }
        Err(e) => {
            println!("Unknown Error, unable to retry: {}", e);
            println!("{:?}", result);
        }
    };

    Ok(())
}
