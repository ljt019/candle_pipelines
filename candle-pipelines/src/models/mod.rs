// ============ Model capability traits (crate-internal) ============

pub(crate) mod capabilities;

// ============ Model implementations ============

pub(crate) mod gemma3;
pub(crate) mod llama3_2;
pub(crate) mod modernbert;
pub(crate) mod olmo3;
pub(crate) mod qwen3;

// Public model config enums (what users see and use)
pub use gemma3::Gemma3;
pub use llama3_2::Llama3_2;
pub use modernbert::ModernBertSize;
pub use olmo3::Olmo3;
pub use qwen3::Qwen3;
