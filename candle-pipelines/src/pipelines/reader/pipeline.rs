use super::model::ReaderModel;
use crate::error::{PipelineError, Result};
use crate::pipelines::text_generation::params::{
    apply_repeat_penalty, initialize_logits_processor, GenerationParams,
};
use candle_core::Tensor;
use futures::Stream;
use tokenizers::Tokenizer;

/// Output format for HTML conversion.
#[derive(Debug, Clone, Copy, Default)]
pub enum OutputFormat {
    /// Convert to clean Markdown.
    #[default]
    Markdown,
    /// Convert to JSON with optional schema.
    Json,
}

/// Pipeline for converting HTML to Markdown or JSON.
///
/// Powered by ReaderLM-v2, a model fine-tuned specifically for HTML parsing.
///
/// # Example
///
/// ```rust,no_run
/// use candle_pipelines::reader::ReaderPipelineBuilder;
///
/// # async fn example() -> candle_pipelines::error::Result<()> {
/// let pipeline = ReaderPipelineBuilder::new().build().await?;
///
/// let html = "<html><body><h1>Hello</h1><p>World</p></body></html>";
/// let markdown = pipeline.to_markdown(html)?;
/// println!("{}", markdown);
/// # Ok(())
/// # }
/// ```
pub struct ReaderPipeline<M: ReaderModel> {
    pub(crate) model: M,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) gen_params: GenerationParams,
}

impl<M: ReaderModel> ReaderPipeline<M> {
    /// Convert HTML to Markdown.
    pub fn to_markdown(&self, html: &str) -> Result<String> {
        self.convert(html, OutputFormat::Markdown, None)
    }

    /// Convert HTML to JSON.
    ///
    /// Optionally provide a JSON schema to guide extraction.
    pub fn to_json(&self, html: &str, schema: Option<&str>) -> Result<String> {
        self.convert(html, OutputFormat::Json, schema)
    }

    /// Convert HTML to Markdown with streaming output.
    ///
    /// Returns a stream that yields text chunks as they're generated.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use candle_pipelines::reader::ReaderPipelineBuilder;
    /// use futures::StreamExt;
    ///
    /// # async fn example() -> candle_pipelines::error::Result<()> {
    /// let pipeline = ReaderPipelineBuilder::new().build().await?;
    /// let html = "<html><body><h1>Hello</h1></body></html>";
    ///
    /// let mut stream = pipeline.to_markdown_stream(html)?;
    /// while let Some(chunk) = stream.next().await {
    ///     print!("{}", chunk?);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn to_markdown_stream(
        &self,
        html: &str,
    ) -> Result<impl Stream<Item = Result<String>> + '_> {
        self.convert_stream(html, OutputFormat::Markdown, None)
    }

    /// Convert HTML to JSON with streaming output.
    pub fn to_json_stream(
        &self,
        html: &str,
        schema: Option<&str>,
    ) -> Result<impl Stream<Item = Result<String>> + '_> {
        self.convert_stream(html, OutputFormat::Json, schema)
    }

    /// Internal conversion method.
    fn convert(&self, html: &str, format: OutputFormat, schema: Option<&str>) -> Result<String> {
        let prompt = self.build_prompt(html, format, schema);

        let input_ids = self
            .tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| PipelineError::Tokenization(e.to_string()))?;

        let input_ids = input_ids.get_ids();

        // Check input length
        if input_ids.len() > self.model.max_seq_len() - self.gen_params.max_len {
            return Err(PipelineError::Unexpected(format!(
                "Input too long: {} tokens (max: {})",
                input_ids.len(),
                self.model.max_seq_len() - self.gen_params.max_len
            )));
        }

        // EOS tokens from generation_config.json: [151645, 151643]
        // 151645 = <|im_end|>, 151643 = <|endoftext|>
        let eos_tokens: &[u32] = &[151645, 151643];

        // Generate with sampling
        let generated_tokens = self.generate_with_sampling(input_ids, eos_tokens)?;

        // Decode output (filter EOS tokens)
        let filtered: Vec<u32> = generated_tokens
            .into_iter()
            .filter(|t| !eos_tokens.contains(t))
            .collect();

        let output = self
            .tokenizer
            .decode(&filtered, true)
            .map_err(|e| PipelineError::Tokenization(e.to_string()))?;

        Ok(output.trim().to_string())
    }

    /// Internal streaming conversion method.
    fn convert_stream(
        &self,
        html: &str,
        format: OutputFormat,
        schema: Option<&str>,
    ) -> Result<impl Stream<Item = Result<String>> + '_> {
        let prompt = self.build_prompt(html, format, schema);

        let encoding = self
            .tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| PipelineError::Tokenization(e.to_string()))?;

        let input_ids = encoding.get_ids().to_vec();

        // Check input length
        if input_ids.len() > self.model.max_seq_len() - self.gen_params.max_len {
            return Err(PipelineError::Unexpected(format!(
                "Input too long: {} tokens (max: {})",
                input_ids.len(),
                self.model.max_seq_len() - self.gen_params.max_len
            )));
        }

        let stream = async_stream::try_stream! {
            let device = self.model.device();
            let params = &self.gen_params;

            // EOS tokens
            let eos_tokens: [u32; 2] = [151645, 151643];

            let mut logits_processor = initialize_logits_processor(params, params.seed);
            let mut generated_tokens: Vec<u32> = Vec::with_capacity(params.max_len);

            // Incremental decoder for streaming
            let mut decoder = self.tokenizer.decode_stream(false);

            // Initial forward pass with full input
            let input_tensor = Tensor::new(input_ids.as_slice(), device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input_tensor, 0)?;
            let logits = logits.squeeze(0)?;

            // Sample first token
            let mut next_token = logits_processor.sample(&logits)?;
            if eos_tokens.contains(&next_token) {
                return;
            }
            generated_tokens.push(next_token);

            // Decode and yield first chunk
            if let Some(chunk) = decoder.step(next_token)
                .map_err(|e| PipelineError::Tokenization(format!("Decode error: {}", e)))? {
                yield chunk;
            }

            let mut position = input_ids.len();

            // Continue generating
            for _ in 1..params.max_len {
                let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
                let logits = self.model.forward(&input, position)?;
                let logits = logits.squeeze(0)?;
                position += 1;

                // Apply repeat penalty
                let logits = if params.repeat_penalty > 1.0 && !generated_tokens.is_empty() {
                    let start_at = generated_tokens.len().saturating_sub(params.repeat_last_n);
                    let penalty_context = &generated_tokens[start_at..];
                    apply_repeat_penalty(&logits, params.repeat_penalty, penalty_context)?
                } else {
                    logits
                };

                next_token = logits_processor.sample(&logits)?;

                if eos_tokens.contains(&next_token) {
                    break;
                }
                generated_tokens.push(next_token);

                // Decode and yield chunk
                if let Some(chunk) = decoder.step(next_token)
                    .map_err(|e| PipelineError::Tokenization(format!("Decode error: {}", e)))? {
                    yield chunk;
                }
            }
        };

        Ok(stream)
    }

    /// Generate tokens with sampling support.
    fn generate_with_sampling(&self, input_ids: &[u32], eos_tokens: &[u32]) -> Result<Vec<u32>> {
        let device = self.model.device();
        let params = &self.gen_params;

        let mut logits_processor = initialize_logits_processor(params, params.seed);
        let mut generated_tokens: Vec<u32> = Vec::with_capacity(params.max_len);

        // Initial forward pass with full input
        let input_tensor = Tensor::new(input_ids, device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input_tensor, 0)?;
        let logits = logits.squeeze(0)?;

        // Sample first token
        let mut next_token = logits_processor.sample(&logits)?;
        if eos_tokens.contains(&next_token) {
            return Ok(generated_tokens);
        }
        generated_tokens.push(next_token);

        let mut position = input_ids.len();

        // Continue generating
        for _ in 1..params.max_len {
            let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, position)?;
            let logits = logits.squeeze(0)?;
            position += 1;

            // Apply repeat penalty
            let logits = if params.repeat_penalty > 1.0 && !generated_tokens.is_empty() {
                let start_at = generated_tokens.len().saturating_sub(params.repeat_last_n);
                let penalty_context = &generated_tokens[start_at..];
                apply_repeat_penalty(&logits, params.repeat_penalty, penalty_context)?
            } else {
                logits
            };

            next_token = logits_processor.sample(&logits)?;

            if eos_tokens.contains(&next_token) {
                break;
            }
            generated_tokens.push(next_token);
        }

        Ok(generated_tokens)
    }

    /// Build the prompt for the model.
    fn build_prompt(&self, html: &str, format: OutputFormat, schema: Option<&str>) -> String {
        let system_msg = match format {
            OutputFormat::Markdown => {
                "You are a helpful assistant that converts HTML to clean, well-formatted Markdown. \
                 Extract the main content and preserve the document structure."
            }
            OutputFormat::Json => {
                "You are a helpful assistant that extracts structured data from HTML \
                 and returns it as valid JSON."
            }
        };

        let user_msg = match (format, schema) {
            (OutputFormat::Markdown, _) => {
                format!("Convert the following HTML to Markdown:\n\n{}", html)
            }
            (OutputFormat::Json, Some(s)) => {
                format!(
                    "Extract data from the following HTML according to this schema:\n{}\n\nHTML:\n{}",
                    s, html
                )
            }
            (OutputFormat::Json, None) => {
                format!(
                    "Extract the main content from this HTML as JSON:\n\n{}",
                    html
                )
            }
        };

        // Qwen2/ChatML format
        format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            system_msg, user_msg
        )
    }

    /// Returns the device the model is running on.
    pub fn device(&self) -> &candle_core::Device {
        self.model.device()
    }
}
