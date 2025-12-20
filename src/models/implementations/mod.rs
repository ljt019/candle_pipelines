pub mod gemma3_candle;
/// Backwards-compatible module path for Gemma3.
///
/// The implementation lives in `gemma3_candle` (to make the backend explicit), but we keep a
/// `gemma3` module so downstream code can continue importing
/// `transformers::models::implementations::gemma3::*`.
pub mod gemma3 {
    pub use super::gemma3_candle::*;
}
pub mod modernbert;
pub mod qwen3;
pub mod qwen3_embeddings;
pub mod qwen3_reranker;

pub use gemma3_candle::{Gemma3Model, Gemma3Size};
pub use modernbert::{ModernBertModel, ModernBertSize};
pub use qwen3::{Qwen3Model, Qwen3Size};
pub use qwen3_embeddings::{Qwen3EmbeddingModel, Qwen3EmbeddingSize};
pub use qwen3_reranker::{Qwen3RerankModel, Qwen3RerankSize};
