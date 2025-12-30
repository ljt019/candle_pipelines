// ============ Model capability traits ============

pub mod capabilities;

#[allow(unused_imports)]
pub use capabilities::{
    // Encoder model traits
    FillMaskModel,
    FillMaskPrediction,
    LabelScores,
    // Text generation traits
    ModelCache,
    ParseEvent,
    Reasoning,
    SentimentAnalysisModel,
    SentimentResult,
    TextGenerationModel,
    ToggleableReasoning,
    ToolCallError,
    ToolCallInvocation,
    ToolCallParser,
    ToolCalling,
    ZeroShotClassificationModel,
};

// ============ Model implementations ============

pub(crate) mod gemma3;
pub(crate) mod llama3_2;
pub(crate) mod modernbert;
pub(crate) mod olmo3;
pub(crate) mod qwen3;

pub use gemma3::{Gemma3, Gemma3Size};
pub use llama3_2::{Llama3_2, Llama3_2Size};
pub use modernbert::ModernBertSize;
pub use olmo3::{Olmo3, Olmo3Size};
pub use qwen3::{Qwen3, Qwen3Size};

// Re-export parsers for advanced use cases
#[allow(unused_imports)]
pub use llama3_2::{
    extract_tool_calls as extract_llama_tool_calls, LlamaToolCall, LlamaToolParser,
};
#[allow(unused_imports)]
pub use olmo3::{extract_tool_calls as extract_olmo3_tool_calls, Olmo3Parser, Olmo3ToolCall};
#[allow(unused_imports)]
pub use qwen3::Qwen3Parser;
