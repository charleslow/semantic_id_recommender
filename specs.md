# Semantic ID Recommender - Technical Specification

## Project Overview

A POC for deploying an LLM-based recommender using semantic IDs:
1. **RQ-VAE** learns semantic IDs for catalogue items (using `vector-quantize-pytorch`)
2. **Small LLM (3B-8B)** is fine-tuned to predict semantic IDs (using `unsloth`)
3. **Constrained generation** at inference ensures valid semantic ID outputs

---

## Key Questions & Recommendations

### 1. Can we train on HuggingFace?

**Yes, with caveats.**

**Options:**
| Platform | GPU Access | Best For |
|----------|-----------|----------|
| HuggingFace Spaces | Limited (T4 free, A10G paid) | Demos, small experiments |
| [HuggingFace AutoTrain](https://huggingface.co/docs/autotrain) | Managed | No-code fine-tuning |
| HuggingFace + External Compute | Full control | Production training |

**Recommendation:** Use HuggingFace Hub for model/data storage, but train on dedicated GPU infrastructure:
- **Modal** or **RunPod** for cost-effective GPU access
- Local GPU if available (RTX 4090/5090 can handle 3B-8B with QLoRA)

**References:**
- [Fine-tune LLMs in 2025 with HuggingFace](https://www.philschmid.de/fine-tune-llms-in-2025)
- [HuggingFace Skills for Claude - Training Integration](https://huggingface.co/blog/hf-skills-training)

---

### 2. Where do we store the model?

**Recommended: HuggingFace Hub**

| Feature | Details |
|---------|---------|
| Free storage | Generous for public models |
| Private storage | 1TB per seat (Team/Enterprise) |
| File size limit | 500GB max per file, recommend <200GB chunks |
| New feature | Xet storage (default since May 2025) - byte-level deduplication |

**Workflow:**
1. Train model locally/cloud
2. Push to HuggingFace Hub (private repo)
3. Pull from Hub during deployment for caching

**Alternatives:**
- AWS S3 / GCS for raw storage
- Modal Volumes for deployment caching

**References:**
- [HuggingFace Storage Limits](https://huggingface.co/docs/hub/en/storage-limits)
- [HuggingFace Pricing](https://huggingface.co/pricing)

---

### 3. Where can we deploy as serverless (<5s cold start, <1s hot)?

**Recommended: Modal or RunPod**

| Platform | Cold Start | Hot Latency | Notes |
|----------|------------|-------------|-------|
| **Modal** | 2-4s | <1s | Best DX, Python-native, good vLLM integration |
| **RunPod** | <200ms (FlashBoot) | <1s | Best cold starts, requires more setup |
| Replicate | 60s+ (custom) | <1s | Only good for pre-built models |

**Optimization strategies for cold start:**
1. Use Modal Volumes to cache HuggingFace models (avoids re-download)
2. Set `FAST_BOOT=True` in vLLM config
3. Use quantized models (4-bit) for faster loading
4. Pre-warm with scheduled pings if needed

**References:**
- [Modal vLLM Deployment Guide](https://modal.com/blog/how-to-deploy-vllm)
- [RunPod FlashBoot](https://www.runpod.io/blog/introducing-flashboot-serverless-cold-start)
- [Top Serverless GPU Clouds 2025](https://www.runpod.io/articles/guides/top-serverless-gpu-clouds)

---

### 4. What base model should we use?

**Recommended: Qwen3-4B or Ministral-3B**

| Model | Size | Why |
|-------|------|-----|
| **Qwen3-4B** | 4B | Best balance of capability/speed, excellent Unsloth support, 128K context |
| **Ministral-3B** | 3B | Smallest viable option, native function calling, good for constrained generation |
| Qwen3-8B | 8B | If more capacity needed, still fits single GPU |
| Llama 3.2-3B | 3B | Alternative, well-documented |

**Key considerations for semantic ID generation:**
- Need strong structured output / constrained generation support
- Ministral-3B has native JSON output capabilities
- vLLM + XGrammar provides efficient constrained decoding for any model

**Unsloth-compatible models:** All above models are supported. See [Unsloth Model Catalog](https://docs.unsloth.ai/get-started/unsloth-model-catalog).

**Constrained generation approach:**
- Use vLLM with `xgrammar` backend (default, fastest)
- Define grammar for valid semantic ID tokens
- Alternatively use `outlines` for regex-based constraints

**References:**
- [Unsloth Model Guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/what-model-should-i-use)
- [Structured Decoding in vLLM](https://blog.vllm.ai/2025/01/14/struct-decode-intro.html)

---

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │  Catalogue   │───▶│   RQ-VAE     │───▶│  Semantic IDs    │   │
│  │  Embeddings  │    │  (vector-    │    │  (codebook       │   │
│  │  (text/item) │    │   quantize)  │    │   indices)       │   │
│  └──────────────┘    └──────────────┘    └──────────────────┘   │
│                                                   │               │
│                                                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │  Training    │───▶│   Unsloth    │───▶│  Fine-tuned LLM  │   │
│  │  Data        │    │   SFT        │    │  (Qwen3-4B)      │   │
│  │  (user,item) │    │              │    │                  │   │
│  └──────────────┘    └──────────────┘    └──────────────────┘   │
│                                                   │               │
│                                                   ▼               │
│                                          ┌──────────────────┐   │
│                                          │  HuggingFace Hub │   │
│                                          │  (model storage) │   │
│                                          └──────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       INFERENCE PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │   User       │───▶│  Modal/      │───▶│   vLLM +         │   │
│  │   Request    │    │  RunPod      │    │   XGrammar       │   │
│  │              │    │  Serverless  │    │   (constrained)  │   │
│  └──────────────┘    └──────────────┘    └──────────────────┘   │
│                                                   │               │
│                                                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │   Item       │◀───│  Semantic ID │◀───│  Generated       │   │
│  │   Lookup     │    │  to Item Map │    │  Semantic IDs    │   │
│  └──────────────┘    └──────────────┘    └──────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: RQ-VAE Training
- Implement RQ-VAE encoder using `vector-quantize-pytorch`
- Train on catalogue item embeddings
- Output: Codebook + semantic ID mapping

### Phase 2: LLM Fine-tuning
- Prepare training data (user history → semantic IDs)
- Fine-tune Qwen3-4B with Unsloth (QLoRA)
- Push to HuggingFace Hub

### Phase 3: Deployment
- Set up Modal serverless endpoint
- Configure vLLM with constrained decoding
- Implement semantic ID → item lookup

### Phase 4: Frontend
- Simple UI for recommendations
- API integration

---

## Cost Estimates (approximate)

| Component | Cost |
|-----------|------|
| Training (Modal A100) | ~$2-5/hour |
| Inference (Modal A10G) | ~$0.60/hour (idle: $0) |
| HuggingFace Hub | Free (public) / ~$9/mo (Pro) |
| Total POC | ~$50-100 |

---

## Project Configuration (Confirmed)

| Parameter | Value | Implications |
|-----------|-------|--------------|
| **Catalogue size** | Medium (1K-100K items) | Standard RQ-VAE with 2-4 codebook levels |
| **Training data** | Cold-start (no interaction logs) | Content-based approach, synthetic training |
| **Deployment** | Modal | 2-4s cold starts, Python-native, vLLM integration |

### Cold-Start Strategy

Since there's no user-item interaction data, we'll use a **content-based approach**:

1. **Semantic ID Assignment:** RQ-VAE encodes item content embeddings → semantic IDs
2. **Training Data Generation:**
   - Use item metadata/descriptions as "queries"
   - Target = semantic ID of matching item
   - Can augment with synthetic user queries via LLM
3. **Inference:** User query → model generates semantic ID → lookup item
