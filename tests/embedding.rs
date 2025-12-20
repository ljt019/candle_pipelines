//! Integration tests for embedding pipeline
//! Run with: cargo test --features integration

#![cfg(feature = "integration")]

use transformers::pipelines::embedding_pipeline::*;
use transformers::pipelines::utils::{BasePipelineBuilder, DeviceSelectable};

#[tokio::test]
async fn embedding_basic() -> anyhow::Result<()> {
    let pipeline = EmbeddingPipelineBuilder::qwen3(Qwen3EmbeddingSize::Size0_6B)
        .cuda_device(0)
        .build()
        .await?;

    let emb = pipeline.embed("hello world").await?;
    assert!(!emb.is_empty());

    let doc_embs = pipeline.embed_batch(&["hello there", "goodbye"]).await?;
    let top = EmbeddingPipeline::<Qwen3EmbeddingModel>::top_k(&emb, &doc_embs, 1);
    assert_eq!(top.len(), 1);
    Ok(())
}

#[tokio::test]
async fn embedding_batch() -> anyhow::Result<()> {
    let pipeline = EmbeddingPipelineBuilder::qwen3(Qwen3EmbeddingSize::Size0_6B)
        .cuda_device(0)
        .build()
        .await?;

    let inputs = ["hello", "world"];
    let embs = pipeline.embed_batch(&inputs).await?;
    assert_eq!(embs.len(), inputs.len());
    for emb in embs {
        assert!(!emb.is_empty());
    }
    Ok(())
}

