//! Llama 3.x tool call parser for streaming detection.
//!
//! Llama outputs raw JSON tool calls: `{"name": "func", "parameters": {...}}`
//! This parser uses a state machine to buffer potential tool calls during streaming
//! and detect them via brace-counting and JSON validation.

use std::collections::VecDeque;

use serde::Deserialize;

use crate::models::capabilities::{ParseEvent, ToolCallInvocation, ToolCallParser};

/// A parsed Llama tool call (internal format).
#[derive(Debug, Clone, Deserialize)]
pub struct LlamaToolCall {
    pub name: String,
    pub parameters: serde_json::Value,
}

impl LlamaToolCall {
    /// Convert to the standard ToolCallInvocation format.
    pub fn into_invocation(self) -> ToolCallInvocation {
        ToolCallInvocation::new(self.name, self.parameters)
    }
}

/// Internal events from the brace-counting state machine.
#[derive(Debug, Clone)]
enum InternalEvent {
    /// Regular content to stream to user.
    Content(String),
    /// A complete tool call was detected.
    ToolCall(LlamaToolCall),
    /// Buffered content wasn't a tool call, flush as regular content.
    Flush(String),
}

/// Parser state for streaming tool call detection.
#[derive(Debug, Clone, Default)]
enum ParseState {
    /// Normal streaming mode - pass tokens through.
    #[default]
    Streaming,
    /// Saw `{`, buffering tokens to determine if it's a tool call.
    Buffering,
}

/// Streaming parser for Llama tool calls.
///
/// Buffers tokens when a potential tool call is detected (`{`),
/// then either emits a `ToolCall` event or flushes the buffer as content.
///
/// Implements [`ToolCallParser`] for integration with the generic tool-aware iterator.
#[derive(Debug, Clone, Default)]
pub struct LlamaToolParser {
    state: ParseState,
    buffer: String,
    brace_depth: i32,
    /// Queue of events to emit (FIFO)
    pending_events: VecDeque<ParseEvent>,
}

impl LlamaToolParser {
    /// Create a new parser.
    pub fn new() -> Self {
        Self::default()
    }

    /// Process a token and return any resulting internal event.
    fn process_token_internal(&mut self, token: &str) -> Option<InternalEvent> {
        match self.state {
            ParseState::Streaming => self.process_streaming(token),
            ParseState::Buffering => self.process_buffering(token),
        }
    }

    /// Finalize parsing when generation ends.
    fn finalize_internal(&mut self) -> Option<InternalEvent> {
        if self.buffer.is_empty() {
            return None;
        }

        let content = std::mem::take(&mut self.buffer);
        self.state = ParseState::Streaming;
        self.brace_depth = 0;

        // Try to parse as tool call
        if let Some(tool_call) = Self::try_parse_tool_call(&content) {
            Some(InternalEvent::ToolCall(tool_call))
        } else {
            Some(InternalEvent::Flush(content))
        }
    }

    fn process_streaming(&mut self, token: &str) -> Option<InternalEvent> {
        let trimmed = token.trim_start();

        // Check if this token starts a potential tool call
        if trimmed.starts_with('{') {
            self.state = ParseState::Buffering;
            self.buffer = token.to_string();
            self.brace_depth = Self::count_braces(token);

            // If braces already balanced, try parsing immediately
            if self.brace_depth == 0 {
                return self.try_complete_buffer();
            }

            None // Buffer, don't emit
        } else {
            Some(InternalEvent::Content(token.to_string()))
        }
    }

    fn process_buffering(&mut self, token: &str) -> Option<InternalEvent> {
        self.buffer.push_str(token);
        self.brace_depth += Self::count_braces(token);

        if self.brace_depth == 0 {
            // Braces balanced - JSON might be complete
            self.try_complete_buffer()
        } else if self.brace_depth < 0 {
            // Invalid nesting - flush as content
            Some(self.flush_buffer())
        } else if self.buffer.len() > 4096 {
            // Too long for a tool call - flush as content
            Some(self.flush_buffer())
        } else {
            None // Keep buffering
        }
    }

    fn try_complete_buffer(&mut self) -> Option<InternalEvent> {
        let content = std::mem::take(&mut self.buffer);
        self.state = ParseState::Streaming;
        self.brace_depth = 0;

        if let Some(tool_call) = Self::try_parse_tool_call(&content) {
            Some(InternalEvent::ToolCall(tool_call))
        } else {
            Some(InternalEvent::Flush(content))
        }
    }

    fn flush_buffer(&mut self) -> InternalEvent {
        let content = std::mem::take(&mut self.buffer);
        self.state = ParseState::Streaming;
        self.brace_depth = 0;
        InternalEvent::Flush(content)
    }

    fn count_braces(s: &str) -> i32 {
        let mut depth = 0i32;
        for c in s.chars() {
            match c {
                '{' => depth += 1,
                '}' => depth -= 1,
                _ => {}
            }
        }
        depth
    }

    fn try_parse_tool_call(content: &str) -> Option<LlamaToolCall> {
        let trimmed = content.trim();

        // Quick check - must start with { and be valid JSON
        if !trimmed.starts_with('{') {
            return None;
        }

        // Try parsing as LlamaToolCall
        let parsed: LlamaToolCall = serde_json::from_str(trimmed).ok()?;

        // Validate it has the expected structure
        if parsed.name.is_empty() {
            return None;
        }

        Some(parsed)
    }

    /// Convert internal event to standard ParseEvent.
    fn convert_event(&self, event: InternalEvent) -> ParseEvent {
        match event {
            InternalEvent::Content(s) => ParseEvent::text(s),
            InternalEvent::ToolCall(call) => ParseEvent::ToolCall(Ok(call.into_invocation())),
            InternalEvent::Flush(s) => {
                // Flushed content that wasn't a valid tool call - emit as text
                ParseEvent::text(s)
            }
        }
    }
}

impl ToolCallParser for LlamaToolParser {
    fn feed(&mut self, token: &str) -> ParseEvent {
        // Process the token
        if !token.is_empty() {
            if let Some(event) = self.process_token_internal(token) {
                self.pending_events.push_back(self.convert_event(event));
            }
        }

        // Return first pending event, or Continue
        self.pending_events
            .pop_front()
            .unwrap_or(ParseEvent::Continue)
    }

    fn flush(&mut self) -> Option<ParseEvent> {
        // First return any pending events
        if let Some(event) = self.pending_events.pop_front() {
            return Some(event);
        }

        // Finalize internal state
        if let Some(event) = self.finalize_internal() {
            return Some(self.convert_event(event));
        }

        None
    }

    fn reset(&mut self) {
        self.state = ParseState::Streaming;
        self.buffer.clear();
        self.brace_depth = 0;
        self.pending_events.clear();
    }
}

/// Extract all tool calls from a complete response (non-streaming).
///
/// For Llama 3.x, tool calls are raw JSON objects with `name` and `parameters`.
pub fn extract_tool_calls(text: &str) -> Vec<LlamaToolCall> {
    let mut tool_calls = Vec::new();
    let trimmed = text.trim();

    // Try parsing the entire response as a single tool call
    if let Ok(call) = serde_json::from_str::<LlamaToolCall>(trimmed) {
        if !call.name.is_empty() {
            tool_calls.push(call);
            return tool_calls;
        }
    }

    // Fall back to scanning for JSON objects
    let mut decoder = JsonObjectScanner::new(trimmed);
    while let Some(json_str) = decoder.next_object() {
        if let Ok(call) = serde_json::from_str::<LlamaToolCall>(json_str) {
            if !call.name.is_empty() {
                tool_calls.push(call);
            }
        }
    }

    tool_calls
}

/// Simple scanner for extracting JSON objects from text.
struct JsonObjectScanner<'a> {
    text: &'a str,
    pos: usize,
}

impl<'a> JsonObjectScanner<'a> {
    fn new(text: &'a str) -> Self {
        Self { text, pos: 0 }
    }

    fn next_object(&mut self) -> Option<&'a str> {
        // Find next '{'
        let start = self.text[self.pos..].find('{')? + self.pos;

        // Count braces to find matching '}'
        let mut depth = 0;
        let mut end = start;

        for (i, c) in self.text[start..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = start + i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }

        if depth != 0 {
            // Unbalanced braces
            self.pos = self.text.len();
            return None;
        }

        self.pos = end;
        Some(&self.text[start..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_tool_call() {
        let json = r#"{"name": "get_weather", "parameters": {"city": "Tokyo"}}"#;
        let calls = extract_tool_calls(json);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
    }

    #[test]
    fn test_streaming_detection() {
        let mut parser = LlamaToolParser::new();

        // Simulate streaming tokens
        let tokens = vec![
            "{",
            "\"name\"",
            ": ",
            "\"test\"",
            ", ",
            "\"parameters\"",
            ": ",
            "{}",
            "}",
        ];

        let mut events = Vec::new();
        for token in tokens {
            let event = parser.feed(token);
            if !event.is_continue() {
                events.push(event);
            }
        }

        // Flush any remaining
        while let Some(event) = parser.flush() {
            events.push(event);
        }

        // Should have one tool call event
        assert!(
            events.iter().any(|e| e.is_tool_call()),
            "Expected ToolCall event, got: {:?}",
            events
        );

        let tool_call = events.iter().find(|e| e.is_tool_call()).unwrap();
        let invocation = tool_call.as_tool_call().unwrap();
        assert_eq!(invocation.name, "test");
    }

    #[test]
    fn test_non_tool_json_flushed() {
        let mut parser = LlamaToolParser::new();

        // JSON without 'name' field should be flushed as text
        let tokens = vec!["{", "\"foo\"", ": ", "\"bar\"", "}"];

        let mut events = Vec::new();
        for token in tokens {
            let event = parser.feed(token);
            if !event.is_continue() {
                events.push(event);
            }
        }

        // Flush any remaining
        while let Some(event) = parser.flush() {
            events.push(event);
        }

        // Should have a text event (flush), not a tool call
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ParseEvent::Text(s) if s.contains("foo"))),
            "Expected Text event containing 'foo', got: {:?}",
            events
        );
        assert!(
            !events.iter().any(|e| e.is_tool_call()),
            "Should not have a tool call event"
        );
    }

    #[test]
    fn test_regular_text_passthrough() {
        let mut parser = LlamaToolParser::new();

        let event = parser.feed("Hello world!");
        assert!(matches!(event, ParseEvent::Text(s) if s == "Hello world!"));
    }

    #[test]
    fn test_reset() {
        let mut parser = LlamaToolParser::new();

        // Start buffering
        parser.feed("{");
        parser.feed("\"name\"");

        // Reset
        parser.reset();

        // Parser should be in clean state
        let event = parser.feed("Hello");
        assert!(matches!(event, ParseEvent::Text(s) if s == "Hello"));
    }
}
