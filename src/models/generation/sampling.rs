//! Logit Processing and Sampling
//!
//! Delegates sampling to Candle's generation utilities while keeping adapters
//! for behaviors we support that Candle does not yet natively expose (e.g.,
//! min-p filtering).

use super::params::GenerationParams;
use candle_core::Tensor;
use candle_transformers::generation::{LogitsProcessor as CandleLogitsProcessor, Sampling};

#[derive(Clone, Debug)]
pub struct LogitsProcessor {
    inner: CandleLogitsProcessor,
    min_p: Option<f64>,
}

impl LogitsProcessor {
    pub fn new(seed: u64, sampling: Sampling, min_p: Option<f64>) -> Self {
        Self {
            inner: CandleLogitsProcessor::from_sampling(seed, sampling),
            min_p,
        }
    }

    pub fn sample(&mut self, logits: &Tensor) -> candle_core::Result<u32> {
        self.sample_f(logits, |_| {})
    }

    pub fn sample_f(
        &mut self,
        logits: &Tensor,
        f: impl FnOnce(&mut [f32]),
    ) -> candle_core::Result<u32> {
        let min_p = self.min_p;
        self.inner.sample_f(logits, |prs| {
            if let Some(min_p) = min_p {
                apply_min_p(prs, min_p as f32);
            }
            f(prs);
        })
    }
}

fn apply_min_p(prs: &mut [f32], min_p: f32) {
    if min_p <= 0.0 || min_p >= 1.0 {
        return;
    }

    let max_prob = prs.iter().copied().fold(0.0f32, f32::max);
    let threshold = min_p * max_prob;

    for p in prs.iter_mut() {
        if *p < threshold {
            *p = 0.0;
        }
    }
}

/// Initializes a LogitsProcessor based on sampling parameters.
pub fn initialize_logits_processor(params: &GenerationParams, seed: u64) -> LogitsProcessor {
    let sampling = params.sampling_strategy();
    LogitsProcessor::new(seed, sampling, params.min_p())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    #[test]
    fn minp_with_min_p_le_0_behaves_like_full_sampling() -> candle_core::Result<()> {
        let dev = candle_core::Device::Cpu;
        let logits = Tensor::from_vec(vec![0.0f32, 0.0, 0.0], 3, &dev)?;
        let mut proc = LogitsProcessor::new(0, Sampling::All { temperature: 1.0 }, None);
        let tok = proc.sample(&logits)?;
        assert!(tok < 3);
        Ok(())
    }
}
