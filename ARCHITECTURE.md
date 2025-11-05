# Architecture Documentation

Detailed technical documentation of the Transformer implementation.

---

## Table of Contents

1. [Model Architecture](#model-architecture)
2. [Component Details](#component-details)
3. [Data Flow](#data-flow)
4. [Training Process](#training-process)
5. [Implementation Decisions](#implementation-decisions)

---

## Model Architecture

### High-Level Structure

The Transformer follows an encoder-decoder architecture:

```
Input Sequence → Encoder → Memory → Decoder → Output Sequence
     ↓                                  ↓
Embeddings +                    Embeddings +
Positional Encoding            Positional Encoding
```

### Layer Stack

**Encoder (N=6 identical layers):**
```
Input
  ↓
Multi-Head Self-Attention
  ↓
Add & Norm (Residual Connection)
  ↓
Position-wise Feed-Forward
  ↓
Add & Norm (Residual Connection)
  ↓
Output
```

**Decoder (N=6 identical layers):**
```
Input
  ↓
Masked Multi-Head Self-Attention
  ↓
Add & Norm
  ↓
Multi-Head Cross-Attention (with Encoder Output)
  ↓
Add & Norm
  ↓
Position-wise Feed-Forward
  ↓
Add & Norm
  ↓
Output
```

---

## Component Details

### 1. Multi-Head Attention (`sublayer.py`)

**Purpose:** Allows the model to attend to different representation subspaces

**Implementation:**
```python
def dot_product_attention(query, key, value, mask=None, dropout=None):
    # Attention(Q, K, V) = softmax(QK^T / √d_k)V
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim_keys)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn
```

**Multi-Head Mechanism:**
- Input dimension: d_model = 512
- Number of heads: h = 8
- Dimension per head: d_k = d_v = 64
- Each head learns different attention patterns
- Outputs are concatenated and linearly projected

**Masking:**
- **Padding mask**: Prevents attention to padding tokens
- **Look-ahead mask**: Prevents decoder from attending to future positions

### 2. Position-wise Feed-Forward Network (`sublayer.py`)

**Architecture:**
```
Input (512) → Linear (2048) → ReLU → Dropout → Linear (512) → Output
```

**Purpose:** Adds non-linearity and capacity to each position independently

**Parameters:**
- Inner dimension: 2048 (4x model dimension)
- Activation: ReLU
- Applied identically to each position

### 3. Positional Encoding (`main.py`)

**Purpose:** Inject sequence order information (Transformers have no inherent ordering)

**Formula:**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Characteristics:**
- Deterministic (not learned)
- Allows model to learn relative positions
- Periodic functions enable extrapolation to longer sequences
- Max length: 5000 tokens

### 4. Layer Normalization (`models.py`)

**Formula:**
```
LayerNorm(x) = γ * (x - μ) / (σ + ε) + β
```

**Implementation Details:**
- Applied before sublayers (pre-norm architecture)
- Normalizes across feature dimension
- Learned parameters: γ (scale), β (bias)
- ε = 1e-6 for numerical stability

### 5. Embeddings (`main.py`)

**Implementation:**
```python
class Embeddings(nn.Module):
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
```

**Scaling Factor:**
- Embeddings are scaled by √d_model = √512 ≈ 22.6
- Balances magnitude with positional encodings
- Prevents positional encodings from dominating

---

## Data Flow

### Training Data Pipeline

1. **Raw Text Files**
   - German: `train.de`, English: `train.en`
   - Parallel corpus (line-aligned)

2. **Preprocessing** (`preprocess.py`)
   - Tokenization with Spacy models
   - Creates JSON format: `{'English': '...', 'German': '...'}`
   - Subsets to first 1000 examples

3. **Torchtext Processing**
   - Field definitions with special tokens: `<s>`, `</s>`, `<blank>`
   - Vocabulary building (max 1000 tokens, min frequency 1)
   - Batching by token count (not sequence count)

4. **Batch Creation**
   - Dynamic batching with `batch_size_fn`
   - Sorts by sequence length
   - Pools 100 batches, sorts, then shuffles
   - Target batch size: 12,000 tokens

### Forward Pass

**Encoder:**
```
Source tokens → Embed → Positional Encoding
    ↓
For each of 6 layers:
    - Self-Attention (attends to all source positions)
    - Add & Norm
    - Feed-Forward
    - Add & Norm
    ↓
Encoder output (memory)
```

**Decoder:**
```
Target tokens → Embed → Positional Encoding
    ↓
For each of 6 layers:
    - Masked Self-Attention (attends to previous target positions only)
    - Add & Norm
    - Cross-Attention (attends to encoder memory)
    - Add & Norm
    - Feed-Forward
    - Add & Norm
    ↓
Linear projection to vocabulary
    ↓
Log-softmax → Output probabilities
```

---

## Training Process

### Optimizer: Noam Scheduler (`optimizer.py`)

**Learning Rate Schedule:**
```python
lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
```

**Parameters:**
- Base learning rate: 0 (schedule controls actual rate)
- Warmup steps: 2000
- Factor: 1
- Model size: 512

**Behavior:**
- Linear warmup for first 2000 steps
- Then inverse square root decay
- Peaks at step 2000, then gradually decreases

### Loss Function: Label Smoothing (`training.py`)

**Purpose:** Regularization technique to prevent overconfidence

**Formula:**
```
y_smoothed = (1 - α) * y_true + α / (K - 1)
```

**Parameters:**
- Smoothing factor α = 0.1
- Target confidence: 0.9 for correct class
- Distributed 0.1 among other classes
- Padding tokens excluded from smoothing

**Loss Computation:**
```python
criterion = nn.KLDivLoss(reduction='sum')
# Computes KL divergence between predicted distribution and smoothed labels
```

### Training Loop

```python
for epoch in range(10):
    model.train()
    for batch in train_iter:
        # Forward pass
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)

        # Compute loss
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        # Backward pass (inside loss_compute)
        # Optimizer step (inside loss_compute)

    # Validation
    model.eval()
    validation_loss = evaluate(model, valid_iter)
```

### Multi-GPU Training

**Strategy:** PyTorch DataParallel

**Implementation:**
```python
if len(devices) > 1:
    model_par = nn.DataParallel(model, device_ids=devices)
else:
    model_par = model
```

**MultiGPULossCompute:**
- Replicates generator across GPUs
- Scatters output and targets
- Computes loss in chunks (chunk_size=5)
- Gathers and aggregates losses
- Backpropagates through original model

---

## Implementation Decisions

### Design Choices

1. **Pre-Norm vs Post-Norm**
   - **Chosen:** Pre-norm (normalize before sublayer)
   - **Rationale:** Better gradient flow, easier training
   - Original paper used post-norm

2. **Attention Dropout**
   - Applied to attention weights after softmax
   - Prevents over-reliance on specific positions
   - Rate: 0.1

3. **Batch Size Strategy**
   - Token-based batching (not sentence-based)
   - Ensures roughly equal computation per batch
   - Implemented via `batch_size_fn`

4. **Vocabulary Size**
   - Limited to 1000 for educational purposes
   - Original datasets have 50K vocabularies
   - Trade-off: faster training vs translation quality

5. **Decoding Strategy**
   - **Implemented:** Greedy decoding
   - Selects argmax at each step
   - **Not implemented:** Beam search
   - Greedy is faster but lower quality

### Known Architectural Differences from Paper

1. **Attention Implementation**
   - Uses `nn.Linear` for projections (standard)
   - Four linear layers: Q, K, V projections + output

2. **Embedding Sharing**
   - Separate embeddings for source and target
   - Paper suggests sharing target embeddings with output layer
   - Not critical for educational purposes

3. **Positional Encoding**
   - Fixed sinusoidal encoding (as in paper)
   - Could use learned positional embeddings

---

## File-by-File Breakdown

### `main.py` (309 lines)
- **Model creation:** `make_model()`
- **Embeddings and positional encoding classes**
- **MultiGPULossCompute:** Distributed training
- **Main training loop**
- **Data loading with torchtext**

### `models.py` (71 lines)
- **EncoderDecoder:** Top-level model wrapper
- **Generator:** Final linear + softmax layer
- **LayerNorm:** Normalization implementation
- **SublayerConnection:** Residual connection wrapper
- **Utility functions:** `multi_layer()`, `subsequent_mask()`

### `encoder.py` (28 lines)
- **Encoder:** Stack of N encoder layers
- **EncoderLayer:** Single encoder layer (attention + FFN)

### `decoder.py` (32 lines)
- **Decoder:** Stack of N decoder layers
- **DecoderLayer:** Single decoder layer (self-attention + cross-attention + FFN)

### `sublayer.py` (62 lines)
- **MultiHeadAttention:** Core attention mechanism
- **PositionWiseFeedForward:** Position-wise FFN
- **dot_product_attention:** Scaled dot-product attention

### `training.py` (152 lines)
- **Batch:** Data structure with masking
- **LabelSmoothing:** Regularized loss
- **SimpleLossCompute:** Single device training
- **MyIterator:** Custom batching logic
- **run_epoch:** Training/validation loop
- **greedy_decode:** Inference function

### `optimizer.py` (64 lines)
- **NoamOpt:** Learning rate scheduler (used)
- **ScheduledOptim:** Alternative scheduler (not used)

### `preprocess.py` (84 lines)
- **Data loading:** Reads parallel text files
- **Tokenization setup:** Spacy models
- **JSON creation:** Prepares data for torchtext

---

## Performance Considerations

### Memory Usage

**Model Parameters:**
```
Total parameters ≈ 44M (for vocab_size=1000)
- Embeddings: 2 * 1000 * 512 = ~1M
- Encoder: 6 layers * 4M = ~24M
- Decoder: 6 layers * 4M = ~24M
- Generator: 512 * 1000 = ~0.5M
```

**Attention Complexity:**
- Time: O(n² * d)
- Space: O(n²) for attention matrices
- n = sequence length, d = model dimension

**Bottlenecks:**
1. Attention computation (quadratic in sequence length)
2. Feed-forward networks (largest parameter count)
3. Vocabulary projection in generator

### Optimization Opportunities

1. **Flash Attention:** Reduce memory for long sequences
2. **Gradient Checkpointing:** Trade compute for memory
3. **Mixed Precision Training:** FP16 for faster training
4. **Batch Size Tuning:** Balance throughput and memory
5. **Efficient Attention:** Sparse or linear attention variants

---

## Testing and Validation

### Manual Testing

**Model Instantiation:**
```python
model = make_model(src_vocab=1000, tgt_vocab=1000, N=6)
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
```

**Inference Test:**
```python
model.eval()
src = torch.LongTensor([[1, 2, 3, 4, 5]])
src_mask = torch.ones(1, 1, 5)
output = greedy_decode(model, src, src_mask, max_len=10, start_symbol=1)
```

### Expected Behavior

- **Untrained model:** Random/gibberish output
- **After training:** Should produce reasonable German translations
- **Validation loss:** Should decrease over epochs

---

## References for Implementation

- Original Paper: Vaswani et al., 2017
- Harvard NLP's "The Annotated Transformer"
- PyTorch nn.Transformer documentation
- fairseq implementation (Facebook AI)

*Last updated: 2025-11-05*
