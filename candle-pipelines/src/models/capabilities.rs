//! Model capability traits.
//!
//! These traits define what models can do - reasoning, tool calling, etc.
//! Pipelines use these traits to orchestrate model behavior.

use crate::error::Result;
use candle_core::{Device, Tensor};
use serde_json::Value;
use tokenizers::Tokenizer;

// ============ Core model traits ============

/// Trait for KV cache types.
pub trait ModelCache: Send {
    /// Reset the cache to empty state.
    fn reset(&mut self);
    /// Get current sequence length in cache.
    fn current_seq_len(&self) -> usize;
}

// ============ Sentiment Analysis ============

/// Internal sentiment result type.
#[derive(Debug, Clone)]
pub struct SentimentResult {
    /// The predicted label.
    pub label: String,
    /// Confidence score.
    pub score: f32,
}

/// Trait for sentiment analysis models.
pub trait SentimentAnalysisModel {
    /// Options type for model configuration.
    type Options: std::fmt::Debug + Clone;

    /// Create a new model instance.
    fn new(options: Self::Options, device: Device) -> Result<Self>
    where
        Self: Sized;

    /// Predict sentiment label for text.
    fn predict(&self, tokenizer: &Tokenizer, text: &str) -> Result<String>;

    /// Predict sentiment for multiple texts.
    fn predict_batch(&self, tokenizer: &Tokenizer, texts: &[&str]) -> Result<Vec<Result<String>>> {
        Ok(texts
            .iter()
            .map(|text| self.predict(tokenizer, text))
            .collect())
    }

    /// Predict sentiment with confidence score.
    fn predict_with_score(&self, tokenizer: &Tokenizer, text: &str) -> Result<SentimentResult> {
        let label = self.predict(tokenizer, text)?;
        Ok(SentimentResult { label, score: 1.0 })
    }

    /// Predict sentiment with score for multiple texts.
    fn predict_with_score_batch(
        &self,
        tokenizer: &Tokenizer,
        texts: &[&str],
    ) -> Result<Vec<Result<SentimentResult>>> {
        Ok(texts
            .iter()
            .map(|text| self.predict_with_score(tokenizer, text))
            .collect())
    }

    /// Get tokenizer for this model.
    fn get_tokenizer(options: Self::Options) -> Result<Tokenizer>;

    /// Get the device this model runs on.
    fn device(&self) -> &Device;
}

// ============ Zero-Shot Classification ============

/// Label-score pairs from zero-shot classification.
pub type LabelScores = Vec<(String, f32)>;

/// Trait for zero-shot classification models.
pub trait ZeroShotClassificationModel {
    /// Options type for model configuration.
    type Options: std::fmt::Debug + Clone;

    /// Create a new model instance.
    fn new(options: Self::Options, device: Device) -> Result<Self>
    where
        Self: Sized;

    /// Classify text against candidate labels.
    fn predict(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        candidate_labels: &[&str],
    ) -> Result<LabelScores>;

    /// Classify multiple texts.
    fn predict_batch(
        &self,
        tokenizer: &Tokenizer,
        texts: &[&str],
        candidate_labels: &[&str],
    ) -> Result<Vec<Result<LabelScores>>> {
        Ok(texts
            .iter()
            .map(|text| self.predict(tokenizer, text, candidate_labels))
            .collect())
    }

    /// Classify with multi-label (independent probabilities).
    fn predict_multi_label(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        candidate_labels: &[&str],
    ) -> Result<LabelScores>;

    /// Multi-label classification for multiple texts.
    fn predict_multi_label_batch(
        &self,
        tokenizer: &Tokenizer,
        texts: &[&str],
        candidate_labels: &[&str],
    ) -> Result<Vec<Result<LabelScores>>> {
        Ok(texts
            .iter()
            .map(|text| self.predict_multi_label(tokenizer, text, candidate_labels))
            .collect())
    }

    /// Get tokenizer for this model.
    fn get_tokenizer(options: Self::Options) -> Result<Tokenizer>;

    /// Get the device this model runs on.
    fn device(&self) -> &Device;
}

// ============ Fill-Mask ============

/// A fill-mask prediction with word and score.
#[derive(Debug, Clone)]
pub struct FillMaskPrediction {
    /// The predicted word.
    pub word: String,
    /// Confidence score.
    pub score: f32,
}

/// Trait for fill-mask models.
pub trait FillMaskModel {
    /// Options type for model configuration.
    type Options: std::fmt::Debug + Clone;

    /// Create a new model instance.
    fn new(options: Self::Options, device: Device) -> Result<Self>
    where
        Self: Sized;

    /// Predict best word for masked position.
    fn predict(&self, tokenizer: &Tokenizer, text: &str) -> Result<String>;

    /// Predict for multiple texts.
    fn predict_batch(&self, tokenizer: &Tokenizer, texts: &[&str]) -> Result<Vec<Result<String>>> {
        Ok(texts
            .iter()
            .map(|text| self.predict(tokenizer, text))
            .collect())
    }

    /// Predict top-k words for masked position.
    fn predict_top_k(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        k: usize,
    ) -> Result<Vec<FillMaskPrediction>> {
        if k == 0 {
            return Ok(vec![]);
        }
        let filled = self.predict(tokenizer, text)?;
        Ok(vec![FillMaskPrediction {
            word: filled.trim().to_string(),
            score: 1.0,
        }])
    }

    /// Predict top-k for multiple texts.
    fn predict_top_k_batch(
        &self,
        tokenizer: &Tokenizer,
        texts: &[&str],
        k: usize,
    ) -> Result<Vec<Result<Vec<FillMaskPrediction>>>> {
        Ok(texts
            .iter()
            .map(|text| self.predict_top_k(tokenizer, text, k))
            .collect())
    }

    /// Get tokenizer for this model.
    fn get_tokenizer(options: Self::Options) -> Result<Tokenizer>;

    /// Get the device this model runs on.
    fn device(&self) -> &Device;
}

// ============ Text Generation ============

/// Core trait for text generation models.
#[allow(async_fn_in_trait)]
pub trait TextGenerationModel {
    /// The options type for model configuration.
    type Options;
    /// The KV cache type for this model.
    type Cache: ModelCache + Send;

    /// Create a new model instance (sync, uses ureq for downloads).
    fn new(options: Self::Options, device: Device) -> Result<Self>
    where
        Self: Sized;

    /// Create a new model instance (async, uses reqwest for downloads).
    async fn new_async(options: Self::Options, device: Device) -> Result<Self>
    where
        Self: Sized;

    /// Get the tokenizer for this model.
    fn get_tokenizer(&self) -> Result<tokenizers::Tokenizer>;

    /// Apply chat template to messages. Tools are included in the prompt if
    /// the model supports tool calling and tools are provided.
    fn apply_chat_template(
        &self,
        messages: &[crate::pipelines::text_generation::Message],
        tools: &[crate::pipelines::text_generation::Tool],
    ) -> Result<String>;

    /// Get the primary end-of-sequence token ID.
    fn get_eos_token(&self) -> Option<u32>;

    /// Get all end-of-sequence token IDs.
    fn get_eos_tokens(&self) -> Vec<u32> {
        self.get_eos_token().into_iter().collect()
    }

    /// Get the maximum sequence length this model supports.
    fn get_max_seq_len(&self) -> usize;

    /// Create a new empty KV cache for generation.
    fn new_cache(&self) -> Self::Cache;

    /// Run forward pass with external cache.
    fn forward(&self, input: &Tensor, cache: &mut Self::Cache) -> candle_core::Result<Tensor>;

    /// Get default generation parameters for this model.
    fn default_generation_params(
        &self,
    ) -> crate::pipelines::text_generation::params::GenerationParams {
        crate::pipelines::text_generation::params::GenerationParams::default()
    }
}

// ============ Reasoning capability ============

/// Marker trait for models that produce reasoning/thinking output.
pub trait Reasoning {}

/// Trait for models where reasoning can be toggled on/off.
pub trait ToggleableReasoning: Reasoning {
    /// Enable or disable reasoning mode.
    fn enable_reasoning(&self, enable: bool);
}

// ============ Tool calling capability ============

/// Trait for models that support tool calling.
///
/// Models implementing this trait can parse tool call formats and generate
/// tool invocations. Tool storage and management is handled by the pipeline.
pub trait ToolCalling {
    /// The parser type for detecting tool calls in this model's output.
    type Parser: ToolCallParser;

    /// Create a new tool call parser for this model.
    fn new_parser(&self) -> Self::Parser;
}

// ============ Tool call parsing ============

/// A successfully parsed tool call invocation.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCallInvocation {
    /// The name of the tool to call.
    pub name: String,
    /// The arguments to pass to the tool.
    pub arguments: Value,
}

impl ToolCallInvocation {
    /// Create a new tool call invocation.
    pub fn new(name: impl Into<String>, arguments: Value) -> Self {
        Self {
            name: name.into(),
            arguments,
        }
    }
}

/// Error information when a tool call fails to parse.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCallError {
    /// The raw content that failed to parse.
    pub raw: String,
    /// The reason parsing failed.
    pub reason: String,
}

impl ToolCallError {
    /// Create a new tool call error.
    pub fn new(raw: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            raw: raw.into(),
            reason: reason.into(),
        }
    }
}

impl std::fmt::Display for ToolCallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Failed to parse tool call: {}", self.reason)
    }
}

impl std::error::Error for ToolCallError {}

/// Events emitted by a tool call parser during streaming.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseEvent {
    /// Normal text output to emit to the user.
    Text(String),

    /// A tool call was detected.
    /// - `Ok(invocation)` if the tool call parsed successfully
    /// - `Err(error)` if the format was recognized but parsing failed
    ToolCall(std::result::Result<ToolCallInvocation, ToolCallError>),

    /// Continue buffering - nothing to emit yet.
    Continue,

    /// An error occurred in the underlying token stream.
    Error(String),
}

impl ParseEvent {
    /// Create a successful tool call event.
    pub fn tool_call(name: impl Into<String>, arguments: Value) -> Self {
        Self::ToolCall(Ok(ToolCallInvocation::new(name, arguments)))
    }

    /// Create a failed tool call event.
    pub fn malformed(raw: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::ToolCall(Err(ToolCallError::new(raw, reason)))
    }

    /// Create a text event.
    pub fn text(content: impl Into<String>) -> Self {
        Self::Text(content.into())
    }

    /// Create an error event.
    pub fn error(message: impl Into<String>) -> Self {
        Self::Error(message.into())
    }

    /// Returns true if this is a Continue event.
    pub fn is_continue(&self) -> bool {
        matches!(self, Self::Continue)
    }

    /// Returns true if this is a successful tool call.
    pub fn is_tool_call(&self) -> bool {
        matches!(self, Self::ToolCall(Ok(_)))
    }

    /// Returns true if this is a malformed tool call.
    pub fn is_malformed(&self) -> bool {
        matches!(self, Self::ToolCall(Err(_)))
    }

    /// Returns true if this is an error event.
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }

    /// Get the tool call invocation if this is a successful tool call.
    pub fn as_tool_call(&self) -> Option<&ToolCallInvocation> {
        match self {
            Self::ToolCall(Ok(invocation)) => Some(invocation),
            _ => None,
        }
    }

    /// Get the error if this is a malformed tool call.
    pub fn as_malformed(&self) -> Option<&ToolCallError> {
        match self {
            Self::ToolCall(Err(error)) => Some(error),
            _ => None,
        }
    }
}

/// Trait for parsing model-specific tool call formats.
///
/// Each model that supports tool calling implements this trait with its
/// specific parsing logic. The parser is stateful and processes tokens
/// one at a time, emitting events when tool calls are detected.
pub trait ToolCallParser: Send {
    /// Feed a token into the parser.
    ///
    /// Returns a [`ParseEvent`] indicating what was detected:
    /// - `Text` - emit this text to the user
    /// - `ToolCall(Ok(...))` - a tool call was detected and parsed
    /// - `ToolCall(Err(...))` - a tool call format was detected but malformed
    /// - `Continue` - keep buffering, nothing to emit yet
    /// - `Error` - an error in the underlying stream
    fn feed(&mut self, token: &str) -> ParseEvent;

    /// Flush any remaining buffered content.
    ///
    /// Called when the token stream ends. Returns any remaining content
    /// that should be emitted (e.g., text that was being buffered).
    fn flush(&mut self) -> Option<ParseEvent>;

    /// Reset the parser state for reuse.
    ///
    /// Called when starting a new generation cycle (e.g., after tool
    /// results are added and generation continues).
    fn reset(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_event_helpers() {
        let text = ParseEvent::text("hello");
        assert!(matches!(text, ParseEvent::Text(s) if s == "hello"));

        let tool = ParseEvent::tool_call("get_weather", serde_json::json!({"city": "Paris"}));
        assert!(tool.is_tool_call());
        assert!(tool.as_tool_call().is_some());

        let malformed = ParseEvent::malformed("<tool_call>bad</tool_call>", "invalid JSON");
        assert!(malformed.is_malformed());
        assert!(malformed.as_malformed().is_some());

        let error = ParseEvent::error("stream failed");
        assert!(error.is_error());

        let cont = ParseEvent::Continue;
        assert!(cont.is_continue());
    }

    #[test]
    fn test_tool_call_invocation() {
        let invocation = ToolCallInvocation::new("test", serde_json::json!({"a": 1}));
        assert_eq!(invocation.name, "test");
        assert_eq!(invocation.arguments, serde_json::json!({"a": 1}));
    }

    #[test]
    fn test_tool_call_error() {
        let error = ToolCallError::new("raw content", "parsing failed");
        assert_eq!(error.raw, "raw content");
        assert_eq!(error.reason, "parsing failed");
        assert_eq!(
            error.to_string(),
            "Failed to parse tool call: parsing failed"
        );
    }
}
