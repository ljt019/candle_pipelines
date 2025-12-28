use candle_core::{Device, Tensor};

/// Trait for models that can perform HTML-to-text conversion.
pub trait ReaderModel {
    /// Maximum sequence length the model supports.
    fn max_seq_len(&self) -> usize;

    /// Device the model is running on.
    fn device(&self) -> &Device;

    /// Forward pass returning logits for the last token.
    fn forward(&self, input_ids: &Tensor, position: usize) -> candle_core::Result<Tensor>;
}

impl ReaderModel for crate::models::readerlm_v2::ReaderLM {
    fn max_seq_len(&self) -> usize {
        crate::models::readerlm_v2::ReaderLM::max_seq_len(self)
    }

    fn device(&self) -> &Device {
        crate::models::readerlm_v2::ReaderLM::device(self)
    }

    fn forward(&self, input_ids: &Tensor, position: usize) -> candle_core::Result<Tensor> {
        crate::models::readerlm_v2::ReaderLM::forward(self, input_ids, position)
    }
}
