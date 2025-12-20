//! Text embedding pipeline for generating dense vector representations of text.
//!
//! This module provides functionality for converting text into high-dimensional
//! vectors that capture semantic meaning, useful for similarity search, clustering,
//! and other downstream tasks.
//!
//! ## Main Types
//!
//! - [`EmbeddingPipeline`] - High-level interface for text embedding
//! - [`EmbeddingPipelineBuilder`] - Builder pattern for pipeline configuration
//! - [`EmbeddingModel`] - Trait for embedding model implementations
//! - [`Qwen3EmbeddingModel`] - Qwen3-based embedding model implementation
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use anyhow::Result;
//! use transformers::pipelines::embedding_pipeline::*;
//! use transformers::pipelines::utils::BasePipelineBuilder;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     // Create an embedding pipeline
//!     let pipeline = EmbeddingPipelineBuilder::qwen3(Qwen3EmbeddingSize::Size0_6B)
//!         .build()
//!         .await?;
//!
//!     // Generate embeddings
//!     let embeddings = pipeline
//!         .embed_batch(&["Hello world", "How are you?"])
//!         .await?;
//!     println!("Generated {} embeddings", embeddings.len());
//!     Ok(())
//! }
//! ```

pub mod builder;
pub mod model;
pub mod pipeline;

pub use builder::EmbeddingPipelineBuilder;
pub use model::EmbeddingModel;
pub use pipeline::EmbeddingPipeline;

pub use crate::models::implementations::qwen3_embeddings::Qwen3EmbeddingModel;
pub use crate::models::implementations::qwen3_embeddings::Qwen3EmbeddingSize;

pub use anyhow::Result;
