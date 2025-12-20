//! # Transformers
//!
//! A Rust library providing pipelines built on top of the
//! [`candle`](https://github.com/huggingface/candle) crate for running
//! large language models locally. See the [README](../README.md) for
//! full details and usage examples.

pub mod core;
pub mod loaders;
pub mod models;
pub mod pipelines;

// Re-export the `#[tool]` procedural macro so users can simply write
// `use transformers::tool;` and annotate their functions without adding an
// explicit dependency on the `tool_macro` crate.
// The macro lives in the separate `tool_macro` crate to avoid a proc-macro/
// normal crate cyclic dependency, but re-exporting it here keeps the public
// API surface of `transformers` ergonomic.

pub use tool_macro::tool;

// Re-export core types
pub use core::{Message, MessageVecExt};

// Re-export model types for easier access
pub use models::implementations::{
    Gemma3Model, Gemma3Size, ModernBertModel, ModernBertSize, Qwen3EmbeddingModel,
    Qwen3EmbeddingSize, Qwen3Model, Qwen3RerankModel, Qwen3Size,
};
