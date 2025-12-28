//! HTML to Markdown/JSON conversion pipeline.
//!
//! Powered by [ReaderLM-v2](https://huggingface.co/jinaai/ReaderLM-v2), a 1.5B parameter
//! model from Jina AI specialized in HTML parsing and conversion.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use candle_pipelines::reader::ReaderPipelineBuilder;
//!
//! # async fn example() -> candle_pipelines::error::Result<()> {
//! let pipeline = ReaderPipelineBuilder::new().build().await?;
//!
//! let html = r#"
//!     <html>
//!         <body>
//!             <h1>Welcome</h1>
//!             <p>This is a <strong>test</strong> document.</p>
//!         </body>
//!     </html>
//! "#;
//!
//! let markdown = pipeline.to_markdown(html)?;
//! println!("{}", markdown);
//! // Output:
//! // # Welcome
//! //
//! // This is a **test** document.
//! # Ok(())
//! # }
//! ```
//!
//! # JSON Extraction
//!
//! Extract structured data with optional schema guidance:
//!
//! ```rust,no_run
//! # use candle_pipelines::reader::ReaderPipelineBuilder;
//! # async fn example() -> candle_pipelines::error::Result<()> {
//! # let pipeline = ReaderPipelineBuilder::new().build().await?;
//! let html = r#"<div class="product"><h2>Widget</h2><span class="price">$19.99</span></div>"#;
//!
//! // With schema
//! let schema = r#"{"name": "string", "price": "string"}"#;
//! let json = pipeline.to_json(html, Some(schema))?;
//! # Ok(())
//! # }
//! ```

// ============ Internal API ============

pub(crate) mod builder;
pub(crate) mod model;
pub(crate) mod pipeline;

// ============ Public API ============

pub use builder::ReaderPipelineBuilder;
pub use pipeline::{OutputFormat, ReaderPipeline};

/// ReaderLM model type. Use [`ReaderPipelineBuilder`] to construct pipelines.
pub type ReaderLM = crate::models::readerlm_v2::ReaderLM;
