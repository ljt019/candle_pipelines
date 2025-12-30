use std::time::{Duration, Instant};

// ============ Encoder stats (batch inference) ============

/// Statistics for encoder model inference (sentiment, fill-mask, zero-shot).
#[derive(Debug, Clone)]
pub struct EncoderStats {
    /// Total execution time.
    pub total_time: Duration,
    /// Number of items processed.
    pub items_processed: usize,
}

impl EncoderStats {
    /// Create a new stats tracker (call at start of operation).
    pub(crate) fn start() -> EncoderStatsBuilder {
        EncoderStatsBuilder {
            start_time: Instant::now(),
        }
    }
}

/// Builder for EncoderStats - tracks timing from creation to finalize.
pub(crate) struct EncoderStatsBuilder {
    start_time: Instant,
}

impl EncoderStatsBuilder {
    /// Finalize stats with the number of items processed.
    pub fn finish(self, items_processed: usize) -> EncoderStats {
        EncoderStats {
            total_time: self.start_time.elapsed(),
            items_processed,
        }
    }
}

// ============ Text generation stats (streaming inference) ============

/// Statistics for text generation, including streaming-specific metrics.
#[derive(Debug, Clone)]
pub struct GenerationStats {
    /// Number of tokens generated.
    pub tokens_generated: usize,
    /// Time until first token was generated (streaming latency).
    pub time_to_first_token: Duration,
    /// Total generation time.
    pub total_time: Duration,
    /// Throughput in tokens per second.
    pub tokens_per_second: f64,
    /// Number of tokens in the prompt.
    pub prompt_tokens: usize,
    start_time: Instant,
    first_token_time: Option<Instant>,
}

impl GenerationStats {
    pub(crate) fn new() -> Self {
        Self {
            tokens_generated: 0,
            time_to_first_token: Duration::default(),
            total_time: Duration::default(),
            tokens_per_second: 0.0,
            prompt_tokens: 0,
            start_time: Instant::now(),
            first_token_time: None,
        }
    }

    pub(crate) fn set_prompt_tokens(&mut self, prompt_tokens: usize) {
        self.prompt_tokens = prompt_tokens;
    }

    pub(crate) fn record_first_token(&mut self) {
        if self.first_token_time.is_none() {
            let now = Instant::now();
            self.first_token_time = Some(now);
            self.time_to_first_token = now.duration_since(self.start_time);
        }
    }

    pub(crate) fn record_token(&mut self) {
        self.tokens_generated += 1;
        if self.first_token_time.is_none() {
            self.record_first_token();
        }
    }

    pub(crate) fn finalize(&mut self) {
        let now = Instant::now();
        self.total_time = now.duration_since(self.start_time);
        if self.total_time.as_secs_f64() > 0.0 {
            self.tokens_per_second = self.tokens_generated as f64 / self.total_time.as_secs_f64();
        } else {
            self.tokens_per_second = 0.0;
        }

        if self.first_token_time.is_none() {
            self.time_to_first_token = Duration::default();
        }
    }

    /// Accumulate stats from another generation (for multi-turn tool calling).
    /// Adds token counts; time_to_first_token is preserved from the first call.
    pub(crate) fn accumulate(&mut self, other: &GenerationStats) {
        self.prompt_tokens += other.prompt_tokens;
        self.tokens_generated += other.tokens_generated;
        // Keep first token time from initial call
    }

    #[cfg(test)]
    pub(crate) fn override_times(
        &mut self,
        start_time: Instant,
        first_token_time: Option<Instant>,
    ) {
        self.start_time = start_time;
        self.first_token_time = first_token_time;
        if let Some(first) = first_token_time {
            self.time_to_first_token = first.duration_since(start_time);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::GenerationStats;
    use std::time::Duration;

    #[test]
    fn calculates_timings_and_rate() {
        let mut stats = GenerationStats::new();
        stats.set_prompt_tokens(5);
        stats.tokens_generated = 4;

        let start = std::time::Instant::now() - Duration::from_secs(2);
        stats.override_times(start, Some(start + Duration::from_millis(500)));

        stats.finalize();

        assert_eq!(stats.prompt_tokens, 5);
        assert_eq!(stats.tokens_generated, 4);
        assert_eq!(stats.time_to_first_token, Duration::from_millis(500));
        assert!(stats.total_time >= Duration::from_secs(2));
        assert!(stats.tokens_per_second > 1.5);
    }
}
