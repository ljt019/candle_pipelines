//! Error types for this crate.
//!
//! All fallible operations return [`Result<T>`] which uses [`TransformersError`] as the error type.

use thiserror::Error;

/// A [`Result`](std::result::Result) alias using [`TransformersError`] as the error type.
pub type Result<T> = std::result::Result<T, TransformersError>;

/// The unified error type for all crate errors.
///
/// # Example
///
/// ```rust,no_run
/// use transformers::error::{TransformersError, Result};
///
/// fn handle_error(e: TransformersError) {
///     match &e {
///         TransformersError::Download(_) => {
///             // Network issue - retry with backoff
///         }
///         TransformersError::Device(_) => {
///             // GPU unavailable - fall back to CPU
///         }
///         TransformersError::Tokenization(_) => {
///             // Bad input - fix and retry
///         }
///         TransformersError::Tool(_) => {
///             // Tool misconfigured - fix tool setup
///         }
///         TransformersError::Unexpected(_) => {
///             // Internal error - report bug
///             eprintln!("Internal error: {e}");
///         }
///     }
/// }
/// ```
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum TransformersError {
    /// Network or download failure. Retry may help.
    #[error("{0}")]
    Download(String),

    /// Tokenization failure. Check input text.
    #[error("{0}")]
    Tokenization(String),

    /// Tool configuration or execution failure. Fix tool setup.
    #[error("{0}")]
    Tool(String),

    /// Device initialization failure. Fall back to CPU.
    #[error("{0}")]
    Device(String),

    /// Internal error. Report if seen.
    #[error("{0}")]
    Unexpected(String),
}

impl From<hf_hub::api::sync::ApiError> for TransformersError {
    fn from(value: hf_hub::api::sync::ApiError) -> Self {
        TransformersError::Download(format!("HuggingFace API error: {}", value))
    }
}

impl From<candle_core::Error> for TransformersError {
    fn from(value: candle_core::Error) -> Self {
        TransformersError::Unexpected(value.to_string())
    }
}

impl From<std::io::Error> for TransformersError {
    fn from(value: std::io::Error) -> Self {
        TransformersError::Unexpected(value.to_string())
    }
}

impl From<serde_json::Error> for TransformersError {
    fn from(value: serde_json::Error) -> Self {
        TransformersError::Unexpected(value.to_string())
    }
}
