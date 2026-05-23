## Building a Tiny LLM from Scratch

This project walks through building a small language model end-to-end from
a bare tokenizer to a live training dashboard. It is structured in three phases.

---

## Building blocks

A GPT-style language model trained on plain text, with a Plotly Dash dashboard
that visualises everything happening during training and loss curves, attention
patterns, token embeddings, and a live text generation panel.

---

### Phase 1: Transformer from scratch

We build the model in PyTorch. This covers a
character-level tokenizer, multi-head causal self-attention with masking, a
feed-forward network, layer normalisation, and the full autoregressive training
loop with AdamW and cosine learning rate decay.


#### 1. Tokenizer — `data/prepare.py`

Before the model sees any text, every character must be mapped to an integer. The so called tokenisation process.

**CharTokenizer** builds a sorted vocabulary of every unique character in the corpus and assigns each one an index:

```
vocab  = {' ': 0, 'T': 1, 'b': 2, 'e': 3, 'o': 4, ...}
"To be" → [1, 4, 0, 2, 3]
```

`TextDataset` turns this flat list into overlapping training pairs. For a window of length T:

```
x = tokens[i   : i + T]      input sequence
y = tokens[i+1 : i + T + 1]  target — x shifted one position right
```

This is the next-token prediction objective: at every position t in x, predict x[t+1]. A model that does this perfectly has learned the full conditional distribution of the language.

**SimpleBPETokenizer** goes further. It iteratively merges the most frequent adjacent character pairs until the vocabulary reaches a target size. After
enough merges, the vocabulary contains sub-word tokens like `"ing"`, `"tion"`, `"the"`, which produces shorter sequences and better generalisation.

---

#### 2. Token and Positional Embeddings

The first thing the model does with an integer token ID is look it up in a
learned embedding table.

**Token embedding:**

```
E_tok ∈ R^(vocab_size × d_model)
tok_emb(x) = E_tok[x]      shape: (B, T, d_model)
```

Each row is a d_model-dimensional vector. Similar tokens end up with similar vectors after training, giving the model semantic understanding.

**Positional embedding:**

The transformer has no inherent sense of order — the same token at position 3 and position 30 would look identical without positional information. We add a
second learned table:

```
E_pos ∈ R^(context_len × d_model)
pos_emb = E_pos[0, 1, 2, ..., T-1]   shape: (T, d_model)
```

The final input to the transformer is their sum:

```
x = dropout( E_tok[tokens] + E_pos[positions] )    shape: (B, T, d_model)
```

Addition in the same vector space lets the model disentangle what a token is
from where it appears.

---

#### 3. Causal Self-Attention

This is the heart of the transformer. When predicting the next word in "The cat sat on the ___", the model should pay more attention to "cat" and "sat" than to "The". Attention learns which tokens matter for each prediction.

**Queries, Keys, and Values**

For each token position we produce three vectors by projecting the input:

```
Q = x · W_Q     shape: (B, T, d_model)
K = x · W_K     shape: (B, T, d_model)
V = x · W_V     shape: (B, T, d_model)
```

In code all three are computed in one shot:

```python
qkv = self.qkv_proj(x)          # (B, T, 3 * d_model)
Q, K, V = qkv.split(d_model, dim=-1)
```

**Scaled dot-product attention**

The attention score between query position i and key position j is their dot product, scaled to keep gradients stable when d_head is large:

```
scores[i, j] = (Q[i] · K[j]) / sqrt(d_head)
```

In matrix form:

```
Attention(Q, K, V) = softmax( Q · K^T / sqrt(d_head) ) · V
```

**Causal mask**

A language model must never look at future tokens. We enforce this by setting all scores where j > i to negative infinity before softmax, so their weight
becomes exactly zero:

```
scores[i, j] = -inf   for all j > i
```

This is the upper-triangular causal mask registered as a buffer in
`CausalSelfAttention`.

**Multi-head attention**

Instead of one attention function over d_model dimensions, we run h independent functions over d_model/h dimensions each:

```
d_head = d_model // n_heads

head_i  = Attention(Q_i, K_i, V_i)      each over d_head dimensions
MultiHead = concat(head_1, ..., head_h) · W_out
```

Each head can specialise — one might attend to syntax, another to semantics. The full shape flow through the layer:

```
x              : (B, T, d_model)
after qkv_proj : (B, T, 3 * d_model)
Q, K, V each   : (B, n_heads, T, d_head)
attn_scores    : (B, n_heads, T, T)
attn_weights   : (B, n_heads, T, T)      softmax over last dim
output         : (B, T, d_model)
```

The attention weights are returned alongside the output so the dashboard can render them as heatmaps without an extra forward pass.

---

#### 4. Feed-Forward Network

After attention, each token position independently passes through a small
two-layer network:

```
FFN(x) = dropout( W_2 · dropout( GELU( W_1 · x ) ) )

W_1 ∈ R^(d_model × d_ff)    expands the dimension (typically d_ff = 4 × d_model)
W_2 ∈ R^(d_ff × d_model) projects back down
```

The expansion and contraction gives the model a large working space for position-wise computation.

**Why GELU instead of ReLU?**

GELU (Gaussian Error Linear Unit) is defined as:

```
GELU(x) = x · Φ(x)
```

where Φ(x) is the CDF of the standard normal distribution. It is smoother than ReLU near zero, which empirically helps language models converge faster. GPT-2, BERT, and virtually all modern transformers use it.

---

#### 5. Layer Normalisation and Residual Connections

**Layer norm** normalises activations across the feature dimension for each token independently. For a vector x of length d:

```
LayerNorm(x) = γ · (x - μ) / √(σ² + ε)  +  β

μ = mean of x across d features
σ = standard deviation across d features
γ, β = learned scale and shift (shape: d)
ε = small constant for numerical stability (1e-5)
```

This project uses **pre-norm** — the norm is applied before each sub-layer,
not after. Pre-norm trains more stably in deep networks because gradients
flow through the residual path without passing through the normalisation.

**Residual connections** wrap every sub-layer:

```
x = x + Attention(LayerNorm(x))     residual 1
x = x + FFN(LayerNorm(x))           residual 2
```

The addition means gradients can bypass sub-layers entirely. Each block learns a small correction on top of its input rather than a full transformation, which is why transformers can be stacked to dozens of layers
without the vanishing gradient problem.

---

#### 6. Language Model Head and Weight Tying

The final layer norm output is projected from d_model back to vocab_size:

```
logits = x · W_lm     W_lm ∈ R^(d_model × vocab_size)
```

These logits go to cross-entropy loss during training, or softmax during generation to produce a probability distribution over the vocabulary.

**Weight tying** sets W_lm equal to the transpose of the token embedding matrix. This halves the number of parameters in the largest layer and consistently improves perplexity. The intuition: the matrix that maps a token into model space should be the same matrix used to map representations back to token space. Used in GPT-2, T5, LLaMA, and almost every modern LLM.

```python
self.lm_head.weight = self.tok_emb.weight   # shared parameter, not a copy
```

---

#### 7. Training Loop — `train/train_scratch.py`

**Objective: cross-entropy loss**

At every token position t the model outputs a probability distribution over the vocabulary. The loss measures how surprised it is by the actual next token:

```
L = - (1/N) Σ log P( x_{t+1} | x_1, ..., x_t )
```

In PyTorch:
```python
F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
```

Every position in every sequence in the batch contributes equally.

**Perplexity** is the exponential of the loss:

```
PPL = exp(L)
```

A perplexity of 10 means the model is as uncertain as if choosing uniformly from 10 equally likely options at each step. Lower is always better.

**AdamW optimiser**

Weights are updated using AdamW — Adam with decoupled weight decay. Adam maintains running estimates of the gradient mean and squared gradient:

```
m_t = β1 · m_{t-1} + (1 - β1) · g_t          first moment (momentum)
v_t = β2 · v_{t-1} + (1 - β2) · g_t²         second moment (adaptive scale)

m̂_t = m_t / (1 - β1^t)                        bias-corrected
v̂_t = v_t / (1 - β2^t)

θ_t = θ_{t-1} - lr · m̂_t / (√v̂_t + ε)  -  lr · λ · θ_{t-1}
```

The last term is weight decay applied directly to the parameters, not to the gradient. Weight decay is only applied to weight matrices, not to biases or layer norm parameters — this is the standard practice from the GPT-2 paper.

**Gradient clipping**

Before each update, the global L2 norm of all gradients is computed:

```
‖g‖ = √( Σ_i g_i² )
```

If this exceeds the clip threshold (1.0 by default), all gradients are scaled
down so the norm equals exactly 1.0. This prevents a single bad batch from
sending the weights to a poor region of parameter space.

---

#### 8. Learning Rate Schedule

Training uses cosine annealing with a linear warmup.

**Linear warmup** (steps 0 → warmup_iters):

```
lr(t) = peak_lr · t / warmup_iters
```

Starting from zero prevents large updates before the optimiser's moment
estimates have stabilised.

**Cosine decay** (steps warmup_iters → lr_decay_iters):

```
progress = (t - warmup_iters) / (lr_decay_iters - warmup_iters)
lr(t) = min_lr + 0.5 · (peak_lr - min_lr) · (1 + cos(π · progress))
```

The cosine shape gives a smooth, gradual reduction so the model keeps
making meaningful updates throughout training rather than stalling early.



---

### Phase 2: Fine-tuning GPT-2 with LoRA — `train/finetune_gpt2.py`

Instead of training from random weights, we take a pre-trained GPT-2 and
adapt it to text using LoRA (Low-Rank Adaptation). Only a small
fraction of the parameters are updated, which means we can run this on a
laptop. 

**What is LoRA?**

Low-Rank Adaptation freezes all original model weights and injects small
trainable matrices alongside the attention projections. For a weight matrix
W ∈ R^(d × k):

```
W' = W + B · A

A ∈ R^(r × k)    initialised with random Gaussian
B ∈ R^(d × r)    initialised with zeros  ->  B·A = 0 at the start of training
```

The rank r is much smaller than min(d, k). For GPT-2 with d = k = 768 and
r = 8, you train 2 × 768 × 8 = 12,288 parameters per layer instead of
768 × 768 = 589,824. Total trainable parameters are typically under 1%
of the full model, which means you can fine-tune on a laptop.

The alpha / r ratio controls the effective scale of each adapted layer.
Setting alpha = 32, r = 8 gives a scale of 4.0.

At inference time the LoRA weights can be merged back into the original
matrices with no extra runtime cost:

```
W_merged = W + (alpha / r) · B · A
```

Only the small adapter files need to be distributed — a few megabytes
rather than gigabytes.


---

### Phase 3:  ML Dashboard— `dashboard/`

`charts.py`

A Dash app reads the training logs written by Phase 1 and 2 and renders them
live as the model trains. Six tabs cover the loss and perplexity curves,
the learning rate schedule, per-layer attention heatmaps, a PCA or t-SNE
scatter of token embeddings, and an interactive generation panel where you
can type a prompt and adjust temperature, top-k, top-p, and beam width in
real time.

`charts.py` contains eight pure Plotly figure builders. None of them import
Dash — they take data and return `go.Figure` objects.

| Chart | What it shows |
|---|---|
| Loss | Train and validation cross-entropy over steps |
| Perplexity | exp(loss) — easier to interpret at a glance |
| LR schedule | The cosine warmup curve as a filled area |
| Gradient norm | Spikes indicate instability; flat near 1.0 is healthy |
| Attention heatmap | (T × T) matrix for a chosen layer and head |
| Embedding scatter | PCA or t-SNE 2D projection of token embedding vectors |
| Token probability bar | Top-k next-token probabilities after a prompt |
| Training overview | 2×2 subplot combining all four training metrics |

`app.py` wraps these in a six-tab Dash layout that auto-refreshes every 10
seconds so you can watch training live in your browser while
`train_scratch.py` runs in another terminal.

`inference_ui.py` provides a generation panel with sliders for temperature,
top-k, top-p, max new tokens, and beam width. Each click calls the model and
renders the output alongside the token probability bar chart.


---

### Generation Strategies — `model/generate.py`

All strategies use the same loop: forward pass -> apply strategy -> sample
next token -> append -> repeat.

**Temperature**

```
logits_scaled = logits / T
probs = softmax(logits_scaled)
```

T < 1.0 sharpens the distribution (more focused).
T > 1.0 flattens it (more creative, more random).

**Top-k sampling**

Keep only the k highest-probability tokens; set the rest to -inf before softmax. Prevents nonsense tokens from ever being sampled.

**Top-p (nucleus) sampling**

Keep the smallest set of tokens whose cumulative probability exceeds p. The nucleus size adapts: small when the model is confident, larger when uncertain. Typical setting: p = 0.9 to 0.95.

**Beam search**

Maintain beam_width candidate sequences. At each step expand every candidate with its top beam_width continuations, then keep the highest-scoring beam_width
sequences by cumulative log-probability:

```
score = Σ_t log P( x_t | x_{<t} )
```

Beam search is deterministic and finds higher-probability sequences than sampling, but is beam_width times more expensive per step.

---



#### Project layout


```
llm_project/
├── data/
│   └── prepare.py          # CharTokenizer, SimpleBPETokenizer, TextDataset, build_dataloaders
├── model/
│   ├── transformer.py      # GPTConfig, CausalSelfAttention, FeedForward, TransformerBlock, GPT
│   └── generate.py         # generate(), beam_search(), get_next_token_probs(), GenerationConfig
├── train/
│   ├── train_scratch.py    # Phase 1 —> full training loop from scratch
│   └── finetune_gpt2.py    # Phase 2 —> LoRA fine-tune GPT-2 (needs transformers + peft)
├── dashboard/
│   ├── app.py              # Dash entrypoint — run this to open the dashboard
│   ├── charts.py           # All Plotly figure builders (pure functions, no Dash)
│   └── inference_ui.py     # Interactive generation panel + callbacks
└── utils/
    ├── logger.py           # TrainingLogger -> training_log.csv
    └── checkpoint.py       # CheckpointManager -> save/resume/best
```


#### For running

**Phase 1: train from scratch**
```bash
python -m train.train_scratch
```

**Dashboard: open while training is running**
```bash
python -m dashboard.app
# http://127.0.0.1:8050
```

**Phase 2: fine-tune GPT-2**
```bash
pip install transformers peft accelerate
python -m train.finetune_gpt2
```

---

#### Dependencies

Phase 1 and the dashboard uses:
`torch`, `numpy`, `pandas`, `scikit-learn`, `plotly`, `dash`.

Phase 2 adds `transformers`, `peft`, and `accelerate` from HuggingFace.

---

#### Optimization

To training on text `train/train_scratch.py`:



Outputs written automatically:
| File | Used by |
|---|---|
| `training_log.csv` | Dashboard loss / ppl / LR charts |
| `attn_weights.npy` | Dashboard attention heatmap |
| `embeddings.npy` | Dashboard PCA / t-SNE scatter |
| `checkpoints/best.pt` | Inference tab |
| `checkpoints/tokenizer.pkl` | Inference tab |

---


Install extra deps (PFET) data sets:

```bash
pip install transformers peft datasets accelerate
```


---

Tabs:
- **Overview:** — 2×2 live training summary
- **Loss:** — train + val loss, LR schedule, grad norm
- **Perplexity:** — perplexity curves
- **Attention:** — per-layer, per-head attention heatmap
- **Embeddings:** — PCA or t-SNE token embedding scatter
- **Inference:** — generate text with temperature / top-k / top-p / beam search sliders

---


```python
CONFIG = {
    "data_path": "path/to/*.txt",    
    ...
}
```

To scale up the model one can adjust:

```python
"n_layers": 6,
"n_heads":  8,
"d_model":  256,
"d_ff":     1024,
```

