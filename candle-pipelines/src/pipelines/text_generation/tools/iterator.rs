//! Generic tool-aware iterator for streaming generation with tool execution.
//!
//! This module provides a single generic iterator that works with any model
//! implementing `TextGenerationModel` and `ToolCalling`. The iterator uses
//! the model's `Parser` type to detect tool calls during streaming.

use std::collections::VecDeque;

use super::Tool;
use crate::error::Result;
use crate::models::capabilities::{
    ParseEvent, TextGenerationModel, ToolCallInvocation, ToolCallParser, ToolCalling,
};
use crate::pipelines::text_generation::message::Message;
use crate::pipelines::text_generation::pipeline::{TextGenerationPipeline, TokenIterator};
use crate::pipelines::stats::GenerationStats;

/// Generic iterator that handles tool calling during streaming generation.
///
/// Works with any model that implements `TextGenerationModel` and `ToolCalling`.
/// Uses the model's `Parser` type to detect tool calls while streaming tokens in real-time.
pub struct GenericToolAwareIterator<'a, M: TextGenerationModel + ToolCalling + Send + Sync> {
    pipeline: &'a TextGenerationPipeline<M>,
    messages: Vec<Message>,
    tools: Vec<Tool>,
    inner: Option<Box<dyn Iterator<Item = Result<String>> + Send + 'a>>,
    parser: <M as ToolCalling>::Parser,
    response_buffer: String,
    pending_output: VecDeque<String>,
    /// Tool calls waiting to be executed
    pending_tool_calls: Vec<ToolCallInvocation>,
    done: bool,
}

impl<'a, M: TextGenerationModel + ToolCalling + Send + Sync> GenericToolAwareIterator<'a, M> {
    /// Create a new tool-aware iterator.
    pub fn new(
        pipeline: &'a TextGenerationPipeline<M>,
        messages: Vec<Message>,
        tools: Vec<Tool>,
        inner: Box<dyn Iterator<Item = Result<String>> + Send + 'a>,
    ) -> Self {
        Self {
            parser: pipeline.model().new_parser(),
            pipeline,
            messages,
            tools,
            inner: Some(inner),
            response_buffer: String::new(),
            pending_output: VecDeque::new(),
            pending_tool_calls: Vec::new(),
            done: false,
        }
    }

    /// Execute pending tool calls and queue results for output.
    fn execute_pending_tools(&mut self) -> Option<Result<String>> {
        if self.pending_tool_calls.is_empty() {
            return None;
        }

        let tool_calls = std::mem::take(&mut self.pending_tool_calls);

        // Add assistant message with the buffered response
        self.messages
            .push(Message::assistant(&self.response_buffer));
        self.response_buffer.clear();
        self.parser.reset();

        // Execute tool calls
        let tool_results = match futures::executor::block_on(
            self.pipeline
                .execute_tool_calls_generic(tool_calls, &self.tools),
        ) {
            Ok(results) => results,
            Err(e) => {
                self.done = true;
                return Some(Err(e));
            }
        };

        // Queue tool result output
        for result in &tool_results {
            self.pending_output
                .push_back(format!("{}\n", result.for_user()));
        }

        // Add raw results as tool messages for model
        for result in tool_results {
            self.messages.push(result.for_model());
        }

        // Create new inner iterator for continued generation
        match self
            .pipeline
            .run_iter_from_messages_owned(self.messages.clone())
        {
            Ok(new_inner) => {
                self.inner = Some(Box::new(new_inner));
            }
            Err(e) => {
                self.done = true;
                return Some(Err(e));
            }
        }

        // Return first pending output
        self.pending_output.pop_front().map(Ok)
    }

    /// Process a parse event from the parser.
    fn process_event(&mut self, event: ParseEvent) -> Option<Result<String>> {
        match event {
            ParseEvent::Text(text) => Some(Ok(text)),
            ParseEvent::ToolCall(Ok(invocation)) => {
                // Queue tool_call XML output
                let json = serde_json::json!({
                    "name": invocation.name,
                    "arguments": invocation.arguments
                });
                self.pending_output.push_back(format!(
                    "<tool_call>\n{}\n</tool_call>\n",
                    serde_json::to_string_pretty(&json).unwrap_or_else(|_| json.to_string())
                ));

                // Store for execution
                self.pending_tool_calls.push(invocation);

                // Return the tool_call output
                self.pending_output.pop_front().map(Ok)
            }
            ParseEvent::ToolCall(Err(error)) => {
                // Malformed tool call - emit as error text so user can filter
                Some(Ok(format!(
                    "<tool_call_error>\n{}\nReason: {}\n</tool_call_error>\n",
                    error.raw, error.reason
                )))
            }
            ParseEvent::Continue => None,
            ParseEvent::Error(msg) => {
                self.done = true;
                Some(Err(crate::error::PipelineError::Unexpected(msg)))
            }
        }
    }
}

impl<M: TextGenerationModel + ToolCalling + Send + Sync> Iterator
    for GenericToolAwareIterator<'_, M>
{
    type Item = Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.done {
                return None;
            }

            // Yield any pending output first
            if let Some(chunk) = self.pending_output.pop_front() {
                return Some(Ok(chunk));
            }

            // If we have pending tool calls and no more output, execute them
            if !self.pending_tool_calls.is_empty() {
                if let Some(result) = self.execute_pending_tools() {
                    return Some(result);
                }
                continue;
            }

            // Try to get next token from inner iterator
            if let Some(ref mut inner) = self.inner {
                if let Some(result) = inner.next() {
                    match result {
                        Ok(token) => {
                            // Buffer for response assembly
                            self.response_buffer.push_str(&token);

                            // Feed to parser
                            let event = self.parser.feed(&token);

                            // Process the event
                            if let Some(output) = self.process_event(event) {
                                return Some(output);
                            }

                            // Continue to next token
                            continue;
                        }
                        Err(e) => return Some(Err(e)),
                    }
                }
                // Inner iterator exhausted
                self.inner = None;
            }

            // Flush parser to get any remaining events
            while let Some(event) = self.parser.flush() {
                if let Some(output) = self.process_event(event) {
                    return Some(output);
                }
            }

            // Execute any remaining tool calls
            if !self.pending_tool_calls.is_empty() {
                if let Some(result) = self.execute_pending_tools() {
                    return Some(result);
                }
                continue;
            }

            // Nothing left
            self.done = true;
            return None;
        }
    }
}

impl<M: TextGenerationModel + ToolCalling + Send + Sync> TokenIterator
    for GenericToolAwareIterator<'_, M>
{
    fn stats(&self) -> GenerationStats {
        GenerationStats::new()
    }
}
