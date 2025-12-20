//! Text reranking pipeline for improving search result relevance.
//!
//! This module provides functionality for reranking a list of documents based on their
//! relevance to a query. It's commonly used in information retrieval systems to improve
//! search result quality by providing more accurate relevance scores.
//!
//! ## Main Types
//!
//! - [`RerankPipeline`] - High-level interface for document reranking
//! - [`RerankPipelineBuilder`] - Builder pattern for pipeline configuration
//! - [`RerankModel`] - Trait for reranking model implementations
//! - [`RerankResult`] - Result containing document and relevance score
//! - [`Qwen3RerankModel`] - Qwen3-based reranking model implementation
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use transformers::pipelines::reranker_pipeline::*;
//! use transformers::pipelines::utils::BasePipelineBuilder;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create a reranking pipeline
//!     let pipeline = RerankPipelineBuilder::qwen3(Qwen3RerankSize::Size0_6B)
//!         .build()
//!         .await?;
//!
//!     // Rerank documents
//!     let query = "machine learning algorithms";
//!     let documents = ["Neural networks", "Decision trees", "Random forests"];
//!     let results = pipeline.rerank(query, &documents).await?;
//!
//!     for result in results {
//!         println!(
//!             "Document: {} (score: {:.3})",
//!             documents[result.index], result.score
//!         );
//!     }
//!     Ok(())
//! }
//! ```

pub mod builder;
pub mod model;
pub mod pipeline;

pub use builder::RerankPipelineBuilder;
pub use model::RerankModel;
pub use pipeline::{RerankPipeline, RerankResult};

pub use crate::models::implementations::qwen3_reranker::{Qwen3RerankModel, Qwen3RerankSize};

pub use anyhow::Result;
