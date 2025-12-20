use candle_core::Device;
use tokenizers::Tokenizer;

/// Base trait for synchronous model implementations used in pipelines.
/// This trait defines the common interface for models that don't require async operations.
pub trait BaseModel {
    /// Configuration options for model initialization
    type Options: std::fmt::Debug + Clone;

    /// Create a new model instance with the given options and device
    fn new(options: Self::Options, device: Device) -> anyhow::Result<Self>
    where
        Self: Sized;

    /// Get a tokenizer configured for this model
    fn get_tokenizer(options: Self::Options) -> anyhow::Result<Tokenizer>;

    /// Get the device this model is running on
    fn device(&self) -> &Device;
}

/// Trait for models that can make single-text predictions
pub trait Predict {
    /// Make a prediction on the given text using the provided tokenizer
    fn predict(&self, tokenizer: &Tokenizer, text: &str) -> anyhow::Result<String>;
}

/// Trait for models that can make multi-class predictions with scores
pub trait PredictWithScores {
    /// Make predictions on the given text, returning labels with their scores
    fn predict(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        labels: &[&str],
    ) -> anyhow::Result<Vec<(String, f32)>>;
}
