# Attention Is All You Need - Transformer Implementation

**CS 547 Semester Project** | Neural Machine Translation (German ↔ English)

🎓 *Educational implementation of the Transformer architecture from the seminal paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)*

---

## 📋 Project Overview

This project implements a **full Transformer encoder-decoder architecture** from scratch using PyTorch for German-to-English neural machine translation. The implementation includes all core components of the Transformer model:

- ✅ Multi-head self-attention mechanism
- ✅ Positional encoding
- ✅ Encoder and decoder stacks (N=6 layers each)
- ✅ Position-wise feed-forward networks
- ✅ Label smoothing regularization
- ✅ Noam optimizer with custom learning rate schedule
- ✅ Multi-GPU training support

---

## 🏗️ Architecture

### Model Components

```
src/
├── main.py          # Main training loop, model creation, data loading
├── models.py        # EncoderDecoder, Generator, LayerNorm, sublayer connections
├── encoder.py       # Encoder and EncoderLayer implementations
├── decoder.py       # Decoder and DecoderLayer implementations
├── sublayer.py      # MultiHeadAttention, PositionWiseFeedForward
├── training.py      # Training utilities, loss computation, batching
└── optimizer.py     # NoamOpt and ScheduledOptim implementations
```

### Key Implementation Details

**Model Hyperparameters:**
- Model dimension (d_model): 512
- Feed-forward dimension (d_ff): 2048
- Number of attention heads: 8
- Number of layers: 6
- Dropout: 0.1
- Vocabulary size: 1000 (configurable)

**Training Configuration:**
- Optimizer: Adam with Noam learning rate schedule
- Label smoothing: 0.1
- Batch size: 12,000 tokens
- Warmup steps: 2,000

---

## 🚀 Setup and Usage

### Prerequisites

```bash
# Install dependencies using pipenv
pipenv install

# Required packages:
- torch
- torchtext
- spacy
- pandas
- tqdm
```

### Spacy Models

```bash
# Download required language models
python -m spacy download en_core_web_trf
python -m spacy download de_dep_news_trf
```

### Data Preparation

1. **Download data** from [Stanford NMT Project](https://nlp.stanford.edu/projects/nmt/)

2. **Organize data structure:**

```
AttentionIsAllYouNeed/
├── src/
├── data/
    ├── test_data/
    │   ├── newstest2014.de
    │   └── newstest2014.en
    ├── train_data/
    │   ├── train.de
    │   └── train.en
    └── vocab_data/
        ├── vocab.50K.de
        └── vocab.50K.en
```

3. **Preprocess data:**

```bash
python preprocess.py  # Generates train.json and test.json
```

### Training

```bash
cd src
python main.py --epoch 10 --batch_size 2048 --n_layers 6
```

**Command-line arguments:**
- `--epoch`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 2048)
- `--d_model`: Model dimension (default: 512)
- `--d_inner_hid`: Feed-forward dimension (default: 2048)
- `--n_head`: Number of attention heads (default: 8)
- `--n_layers`: Number of encoder/decoder layers (default: 6)
- `--dropout`: Dropout rate (default: 0.1)
- `--warmup`: Warmup steps for optimizer (default: 4000)

---

## 🎓 What Was Learned

### Technical Achievements

1. **Attention Mechanism Deep Dive**
   - Implemented scaled dot-product attention from scratch
   - Understanding of multi-head attention parallelization
   - Masking for autoregressive decoding

2. **Modern Deep Learning Practices**
   - Label smoothing for better generalization
   - Custom learning rate schedules (Noam optimizer)
   - Residual connections and layer normalization
   - Multi-GPU training with PyTorch DataParallel

3. **NLP Pipeline Development**
   - Tokenization with Spacy
   - Vocabulary building and management
   - Batching strategies for variable-length sequences
   - Greedy decoding for inference

4. **PyTorch Proficiency**
   - Dynamic computation graphs
   - Custom nn.Module implementations
   - Device management (CPU/GPU)
   - Legacy torchtext API usage

### Key Insights

- **Positional encoding is crucial** - Without it, the Transformer cannot distinguish token order
- **Layer normalization placement** - Pre-norm vs post-norm architectures have different training dynamics
- **Attention is computationally expensive** - O(n²) complexity with sequence length
- **Multi-GPU coordination** - Managing model replication and gradient synchronization

---

## ⚠️ Known Issues and Limitations

### Educational Simplifications

1. **Small vocabulary** - Training uses only 1,000 tokens (vs 50K available)
2. **Limited data** - Preprocessing uses first 1,000 examples only
3. **No checkpointing** - Model state is not saved during training
4. **No validation metrics** - Missing BLEU score calculation
5. **No beam search** - Only greedy decoding implemented

### Technical Debt (Fixed in Final Commit)

The following issues were identified and **fixed** in the final version:

- ✅ **Swapped language models** - German/English spacy models were reversed
- ✅ **Deprecated PyTorch APIs** - Updated `loss.data[0]` → `loss.item()`
- ✅ **Hardcoded CUDA** - Now supports CPU and dynamic GPU detection
- ✅ **Inefficient device transfers** - Removed redundant `.to(device)` calls

### Remaining Limitations

- Uses legacy `torchtext.legacy` API (deprecated in newer PyTorch)
- Uses deprecated `Variable` wrapper (unnecessary since PyTorch 0.4)
- No proper logging infrastructure (uses prints)
- No experiment tracking (no Tensorboard/WandB integration)
- Hard to tune hyperparameters (all hardcoded)

---

## 📊 Project Status: ARCHIVED

This project was completed as a course assignment and is now archived. The code successfully demonstrates:
- ✅ Complete Transformer implementation
- ✅ Working training pipeline
- ✅ Multi-GPU support
- ✅ All core components from the paper

**Not intended for production use** - This is an educational implementation for learning purposes.

---

## 📚 References

### Primary Source
- **Vaswani et al. (2017)** - ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)

### Implementation Guidance
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) by Harvard NLP
- PyTorch official documentation
- [Stanford NLP NMT Data](https://nlp.stanford.edu/projects/nmt/)

### Tools and Libraries
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Spacy](https://spacy.io/) - Tokenization
- [GloVe](https://github.com/stanfordnlp/GloVe) - Word embeddings (referenced)

---

## 📝 License

This project was created for educational purposes as part of CS 547. See LICENSE file for details.

---

## 🙏 Acknowledgments

Thanks to the CS 547 instructors and the original Transformer paper authors for making this learning opportunity possible.

*Last updated: 2025-11-05*
