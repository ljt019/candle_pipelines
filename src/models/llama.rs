use std::io::{Read, Seek};
use std::sync::Arc;

use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_transformers::models::quantized_llama as candle_llama;
use minijinja::UndefinedBehavior;
use minijinja::{context, Environment};
use minijinja_contrib::{add_to_environment, pycompat};
use tokenizers::Tokenizer;

use crate::loaders::GenerationConfig;
use crate::loaders::{GenerationConfigLoader, GgufModelLoader, HfLoader, TokenizerLoader};
use crate::pipelines::text_generation::model::Tool;
use crate::pipelines::text_generation::model::{
    LanguageModelContext, TextGenerationModel, ToolCalling,
};

#[derive(Debug, Clone, Copy)]
pub enum LlamaSize {
    Size1B,
    Size3B,
}

impl LlamaSize {
    pub fn weight_repo_id(&self) -> &str {
        match self {
            LlamaSize::Size1B => "bartowski/Llama-3.2-1B-Instruct-GGUF",
            LlamaSize::Size3B => "bartowski/Llama-3.2-3B-Instruct-GGUF",
        }
    }

    pub fn weight_filename(&self) -> &str {
        match self {
            LlamaSize::Size1B => "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            LlamaSize::Size3B => "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        }
    }

    pub fn config_repo_id(&self) -> &str {
        match self {
            LlamaSize::Size1B => "meta-llama/Llama-3.2-1B-Instruct",
            LlamaSize::Size3B => "meta-llama/Llama-3.2-3B-Instruct",
        }
    }
}

impl std::fmt::Display for LlamaSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            LlamaSize::Size1B => "llama-3.2-1b",
            LlamaSize::Size3B => "llama-3.2-3b",
        };
        write!(f, "{name}")
    }
}

impl crate::pipelines::cache::ModelOptions for LlamaSize {
    fn cache_key(&self) -> String {
        self.to_string()
    }
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub num_layers: usize,
    pub max_seq_len: usize,
    pub dtype: DType,
    pub device: Device,
    pub target_device: Option<String>,
}

#[derive(Clone)]
pub struct LlamaModel {
    base_weights: Arc<candle_llama::ModelWeights>,
    info: ModelInfo,
    tokenizer_repo_id: String,
    generation_config: GenerationConfig,
    chat_template_env: Arc<Environment<'static>>,
    tools: Vec<Tool>,
}

impl LlamaModel {
    fn parse_metadata(content: &gguf_file::Content, device: &Device) -> anyhow::Result<ModelInfo> {
        let num_layers = content
            .metadata
            .get("llama.block_count")
            .and_then(|v| v.to_u32().ok())
            .ok_or_else(|| anyhow::anyhow!("Missing critical metadata: llama.block_count"))?
            as usize;

        let max_seq_len = content
            .metadata
            .get("llama.context_length")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(4096) as usize;

        let dtype = content
            .metadata
            .get("general.dtype")
            .and_then(|v| v.to_u32().ok())
            .and_then(|v| match v {
                0 => Some(DType::F32),
                1 => Some(DType::F16),
                _ => None,
            })
            .unwrap_or(DType::F16);

        let target_device = content
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .cloned();

        Ok(ModelInfo {
            num_layers,
            max_seq_len,
            dtype,
            device: device.clone(),
            target_device,
        })
    }

    async fn load_chat_template_env(
        tokenizer_repo_id: &str,
    ) -> anyhow::Result<Arc<Environment<'static>>> {
        let hf_loader = HfLoader::new(tokenizer_repo_id);
        let tokenizer_config_json = hf_loader.json_file("tokenizer_config.json").await?;

        let chat_template_str = tokenizer_config_json
            .get("chat_template")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'chat_template' field in tokenizer config"))?;

        let mut env = Environment::new();
        env.set_undefined_behavior(UndefinedBehavior::Lenient);
        add_to_environment(&mut env);
        env.set_unknown_method_callback(pycompat::unknown_method_callback);
        env.add_filter("tojson", minijinja::filters::tojson);

        env.add_template_owned("chat", chat_template_str.to_string())?;

        Ok(Arc::new(env))
    }

    fn eos_tokens(&self) -> Vec<u32> {
        self.generation_config
            .eos_token_ids
            .iter()
            .map(|&id| id as u32)
            .collect()
    }

    fn ensure_eos_tokens(config: &GenerationConfig) -> anyhow::Result<()> {
        if config.eos_token_ids.is_empty() {
            anyhow::bail!("Llama generation config is missing 'eos_token_ids'");
        }

        Ok(())
    }

    pub async fn from_gguf<R: Read + Seek>(
        reader: &mut R,
        device: &Device,
        size: LlamaSize,
    ) -> anyhow::Result<Self> {
        let content = gguf_file::Content::read(reader)?;
        let info = Self::parse_metadata(&content, device)?;
        let weights = Arc::new(candle_llama::ModelWeights::from_gguf(
            content, reader, device,
        )?);

        let tokenizer_repo_id = size.config_repo_id().to_string();
        let generation_config =
            GenerationConfigLoader::new(&tokenizer_repo_id, "generation_config.json")
                .load()
                .await?;
        Self::ensure_eos_tokens(&generation_config)?;
        let chat_template_env = Self::load_chat_template_env(&tokenizer_repo_id).await?;

        Ok(Self {
            base_weights: weights,
            info,
            tokenizer_repo_id,
            generation_config,
            chat_template_env,
            tools: Vec::new(),
        })
    }

    pub async fn from_hf(device: &Device, size: LlamaSize) -> anyhow::Result<Self> {
        let loader = GgufModelLoader::new(size.weight_repo_id(), size.weight_filename());
        let (mut file, content) = loader.load().await?;
        let info = Self::parse_metadata(&content, device)?;
        let weights = Arc::new(candle_llama::ModelWeights::from_gguf(
            content, &mut file, device,
        )?);

        let tokenizer_repo_id = size.config_repo_id().to_string();
        let generation_config =
            GenerationConfigLoader::new(&tokenizer_repo_id, "generation_config.json")
                .load()
                .await?;
        Self::ensure_eos_tokens(&generation_config)?;
        let chat_template_env = Self::load_chat_template_env(&tokenizer_repo_id).await?;

        Ok(Self {
            base_weights: weights,
            info,
            tokenizer_repo_id,
            generation_config,
            chat_template_env,
            tools: Vec::new(),
        })
    }

    pub async fn get_tokenizer(&self) -> anyhow::Result<Tokenizer> {
        let tokenizer_loader = TokenizerLoader::new(&self.tokenizer_repo_id, "tokenizer.json");
        tokenizer_loader.load().await
    }

    pub fn create_context(&self) -> Context {
        Context::new((*self.base_weights).clone())
    }
}

pub struct Context {
    base_weights: candle_llama::ModelWeights,
    weights: candle_llama::ModelWeights,
    position: usize,
}

impl Context {
    pub fn new(weights: candle_llama::ModelWeights) -> Self {
        Self {
            base_weights: weights.clone(),
            weights,
            position: 0,
        }
    }

    pub fn reset_with(&mut self, weights: &candle_llama::ModelWeights) {
        self.base_weights = weights.clone();
        self.weights = weights.clone();
        self.position = 0;
    }

    pub fn generate(&mut self, input: &Tensor) -> CandleResult<Tensor> {
        let input = if input.dtype() != DType::I64 {
            input.to_dtype(DType::I64)?
        } else {
            input.clone()
        };
        let seq_len = input.dim(1)? as usize;
        let logits = self.weights.forward(&input, self.position)?;
        self.position += seq_len;
        Ok(logits)
    }
}

impl LanguageModelContext for Context {
    fn generate(&mut self, input: &Tensor) -> candle_core::Result<Tensor> {
        Context::generate(self, input)
    }

    fn reset(&mut self) {
        self.weights = self.base_weights.clone();
        self.position = 0;
    }

    fn position(&self) -> usize {
        self.position
    }

    fn can_continue_from(&self, position: usize) -> bool {
        self.position == position
    }
}

impl TextGenerationModel for LlamaModel {
    type Options = LlamaSize;
    type Context = Context;

    async fn new(options: Self::Options, device: Device) -> anyhow::Result<Self> {
        LlamaModel::from_hf(&device, options).await
    }

    async fn get_tokenizer(&self) -> anyhow::Result<Tokenizer> {
        LlamaModel::get_tokenizer(self).await
    }

    fn apply_chat_template(&self, messages: &[crate::Message]) -> anyhow::Result<String> {
        let messages_dicts: Vec<serde_json::Value> = messages
            .iter()
            .map(|msg| {
                serde_json::json!({
                    "role": msg.role().as_str(),
                    "content": msg.content(),
                })
            })
            .collect();

        let rendered = self
            .chat_template_env
            .get_template("chat")?
            .render(context! {
                messages => messages_dicts,
                add_generation_prompt => true,
                tools => self.registered_tools(),
            })?;
        Ok(rendered)
    }

    fn get_eos_token(&self) -> Option<u32> {
        self.eos_tokens().into_iter().next()
    }

    fn get_eos_tokens(&self) -> Vec<u32> {
        self.eos_tokens()
    }

    fn get_max_seq_len(&self) -> usize {
        self.info.max_seq_len
    }

    fn new_context(&self) -> Context {
        self.create_context()
    }

    fn clear_context(&self, context: &mut Context) -> anyhow::Result<()> {
        context.reset_with(&self.base_weights);
        Ok(())
    }

    fn default_generation_params(
        &self,
    ) -> crate::pipelines::text_generation::params::GenerationParams {
        crate::pipelines::text_generation::params::GenerationParams {
            temperature: self.generation_config.temperature.unwrap_or(0.7),
            repeat_penalty: self.generation_config.repeat_penalty.unwrap_or(1.1),
            repeat_last_n: self.generation_config.repeat_last_n.unwrap_or(64),
            seed: rand::random(),
            max_len: 4096,
            top_p: Some(self.generation_config.top_p.unwrap_or(0.9)),
            top_k: Some(self.generation_config.top_k.unwrap_or(40) as usize),
            min_p: Some(self.generation_config.min_p.unwrap_or(0.0)).filter(|v| *v > 0.0),
        }
    }
}

impl ToolCalling for LlamaModel {
    fn register_tool(&mut self, tool: Tool) -> anyhow::Result<()> {
        if let Some(pos) = self.tools.iter().position(|t| t.name() == tool.name()) {
            self.tools[pos] = tool;
        } else {
            self.tools.push(tool);
        }
        Ok(())
    }

    fn unregister_tool(&mut self, name: &str) -> anyhow::Result<()> {
        if let Some(pos) = self.tools.iter().position(|t| t.name() == name) {
            self.tools.remove(pos);
        }
        Ok(())
    }

    fn clear_tools(&mut self) -> anyhow::Result<()> {
        self.tools.clear();
        Ok(())
    }

    fn registered_tools(&self) -> Vec<Tool> {
        self.tools.clone()
    }

    fn call_tool(
        &mut self,
        tool_name: String,
        parameters: std::collections::HashMap<String, String>,
    ) -> std::result::Result<String, crate::pipelines::text_generation::tools::ToolError> {
        if let Some(tool) = self.tools.iter().find(|t| t.name() == tool_name) {
            tool.call(parameters)
        } else {
            Err(
                crate::pipelines::text_generation::tools::ToolError::Message(format!(
                    "Tool '{tool_name}' is not registered"
                )),
            )
        }
    }
}
