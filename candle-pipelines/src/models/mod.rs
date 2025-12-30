// ============ Model capability traits (crate-internal) ============

pub(crate) mod capabilities;

// ============ Model implementations ============

pub(crate) mod gemma3;
pub(crate) mod llama3_2;
pub(crate) mod modernbert;
pub(crate) mod olmo3;
pub(crate) mod qwen3;

// Public model structs and size enums (for type annotations)
pub use gemma3::{Gemma3, Gemma3Size};
pub use llama3_2::{Llama3_2, Llama3_2Size};
pub use modernbert::ModernBertSize;
pub use olmo3::{Olmo3, Olmo3Size};
pub use qwen3::{Qwen3, Qwen3Size};
