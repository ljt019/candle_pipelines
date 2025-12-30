//! Tool calling infrastructure for text generation.
//!
//! This module provides the runtime components for tool calling:
//! - [`Tool`] - Runtime representation of a callable tool
//! - [`ErrorStrategy`] - How to handle tool execution errors
//! - [`GenericToolAwareIterator`] - Streaming iterator with tool execution

mod iterator;
mod tool;

pub use iterator::GenericToolAwareIterator;
pub use tool::{ErrorStrategy, Tool, ToolFuture};
