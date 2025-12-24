//! Error types for this crate.
//!
//! Most functions return [`Result<T>`] which uses [`TransformersError`] as the error type.

use thiserror::Error;

/// A [`Result`](std::result::Result) alias using [`TransformersError`] as the error type.
pub type Result<T> = std::result::Result<T, TransformersError>;

/// The unified error type for all crate errors.
///
/// # Example
///
/// ```
/// use transformers::error::{TransformersError, DownloadError};
///
/// fn handle_error(e: TransformersError) {
///     match &e {
///         TransformersError::Download(DownloadError::Timeout { .. }) => {
///             // Retry
///         }
///         TransformersError::Device(_) => {
///             // Fall back to CPU
///         }
///         TransformersError::Unexpected(_) => {
///             // Log full chain and report - nothing user can do
///             eprintln!("Internal error: {e:?}");
///         }
///         _ => {}
///     }
/// }
/// ```
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum TransformersError {
    /// Errors during file download. Users may retry or check connectivity.
    #[error(transparent)]
    Download(#[from] DownloadError),

    /// Errors during tokenization. Users may fix malformed input.
    #[error(transparent)]
    Tokenization(#[from] TokenizationError),

    /// Errors when using tools. Users may fix tool definitions or parameters.
    #[error(transparent)]
    Tool(#[from] ToolError),

    /// Errors when initializing a device. Users may fall back to CPU.
    #[error(transparent)]
    Device(#[from] DeviceError),

    /// Internal/unexpected error. Users cannot act on this, report if seen.
    ///
    /// The underlying error chain is preserved via [`std::error::Error::source()`]
    /// for debugging and logging purposes.
    #[error(transparent)]
    Unexpected(Box<dyn std::error::Error + Send + Sync>),
}

/// Errors that can occur during a file download operation.
///
/// Users may be able to recover by retrying, checking network connectivity,
/// or verifying the repository/file names.
#[derive(Error, Debug, Clone)]
#[non_exhaustive]
pub enum DownloadError {
    /// Failed to download a file from a given HF repository.
    #[error("Failed to download '{file}' from '{repo}': {reason}")]
    Failed {
        /// The repository from which the file was being downloaded.
        repo: String,
        /// The file that failed to download.
        file: String,
        /// The reason why the download failed.
        reason: String,
    },

    /// A download operation timed out.
    #[error("Download timed out for '{file}' from '{repo}' after {attempts} attempt(s)")]
    Timeout {
        /// The repository from which the download was attempted.
        repo: String,
        /// The file that could not be downloaded before the timeout.
        file: String,
        /// The number of attempts made before timing out.
        attempts: u32,
    },

    /// Failed to initialize the HuggingFace API.
    #[error("Failed to initialize HuggingFace API: {reason}")]
    ApiInit {
        /// The reason HF returned as to why the API initialization failed.
        reason: String,
    },
}

/// Errors that can occur when tokenizing input text.
///
/// Users may be able to recover by fixing malformed input or checking file paths.
#[derive(Error, Debug, Clone)]
#[non_exhaustive]
pub enum TokenizationError {
    /// Failed to load tokenizer from a given path.
    #[error("Failed to load tokenizer from '{path}': {reason}")]
    LoadFailed {
        /// The path that failed to load the tokenizer.
        path: String,
        /// The reason why the tokenizer failed to load.
        reason: String,
    },

    /// Tokenization failed on a given input.
    #[error("Tokenization failed on '{input_preview}': {reason}")]
    EncodeFailed {
        /// The preview of the input text that failed to encode.
        input_preview: String,
        /// The reason why the tokenization failed to encode.
        reason: String,
    },

    /// Failed to decode a token.
    #[error("Failed to decode token {token_id}: {reason}")]
    DecodeFailed {
        /// The ID of the token that failed to decode.
        token_id: u32,
        /// The reason why the token failed to decode.
        reason: String,
    },
}

impl TokenizationError {
    pub(crate) fn encode_failed(input: &str, reason: impl Into<String>) -> Self {
        let preview: String = input.chars().take(50).collect();
        Self::EncodeFailed {
            input_preview: preview,
            reason: reason.into(),
        }
    }
}

/// Errors that can occur when using tools.
///
/// Users define and register tools, so they can act on these errors
/// by fixing tool definitions, parameters, or registration.
#[derive(Error, Debug, Clone)]
#[non_exhaustive]
pub enum ToolError {
    /// Tool not found.
    #[error("Tool '{name}' not found. Registered tools: {}", .available.join(", "))]
    NotFound {
        /// The name of the tool that was not found.
        name: String,
        /// The available tools.
        available: Vec<String>,
    },

    /// No tools registered.
    #[error("No tools registered. Call register_tools() before completion_with_tools().")]
    NoToolsRegistered,

    /// Tool execution failed.
    #[error("Tool '{name}' failed after {attempts} attempt(s): {reason}")]
    ExecutionFailed {
        /// The name of the tool that failed to execute.
        name: String,
        /// The number of attempts made before the tool execution failed.
        attempts: u32,
        /// The reason why the tool failed to execute.
        reason: String,
    },

    /// Invalid parameters for a tool.
    #[error("Invalid parameters for '{name}': {reason}")]
    InvalidParams {
        /// The name of the tool that has invalid parameters.
        name: String,
        /// The reason why the tool has invalid parameters.
        reason: String,
    },

    /// Schema error for a tool.
    #[error("Schema error for '{name}': {reason}")]
    SchemaError {
        /// The name of the tool that has a schema error.
        name: String,
        /// The reason why the tool has a schema error.
        reason: String,
    },
}

/// Errors that can occur when initializing a device.
///
/// Users can recover by falling back to CPU via [`DeviceRequest::Cpu`].
#[derive(Error, Debug, Clone)]
#[non_exhaustive]
pub enum DeviceError {
    /// Failed to init CUDA device.
    #[error("Failed to init CUDA device {index}: {reason}. Try CPU as fallback.")]
    CudaInitFailed {
        /// The index of the CUDA device that failed to initialize.
        index: usize,
        /// The reason why the CUDA device failed to initialize.
        reason: String,
    },
}

impl From<candle_core::Error> for TransformersError {
    fn from(value: candle_core::Error) -> Self {
        TransformersError::Unexpected(Box::new(value))
    }
}

impl From<std::io::Error> for TransformersError {
    fn from(value: std::io::Error) -> Self {
        TransformersError::Unexpected(Box::new(value))
    }
}

impl From<serde_json::Error> for TransformersError {
    fn from(value: serde_json::Error) -> Self {
        TransformersError::Unexpected(Box::new(value))
    }
}

impl From<hf_hub::api::sync::ApiError> for TransformersError {
    fn from(value: hf_hub::api::sync::ApiError) -> Self {
        DownloadError::ApiInit {
            reason: value.to_string(),
        }
        .into()
    }
}

impl From<Box<dyn std::error::Error + Send + Sync>> for TransformersError {
    fn from(value: Box<dyn std::error::Error + Send + Sync>) -> Self {
        TransformersError::Unexpected(value)
    }
}

impl From<&str> for TransformersError {
    fn from(value: &str) -> Self {
        TransformersError::Unexpected(value.to_string().into())
    }
}

impl From<String> for TransformersError {
    fn from(value: String) -> Self {
        TransformersError::Unexpected(value.into())
    }
}
