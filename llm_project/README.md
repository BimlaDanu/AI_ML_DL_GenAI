### Building a Tiny LLM from Scratch

This project walks through building a small language model end-to-end from
a bare tokenizer to a live training dashboard. It is structured in three phases.

---

#### Building blocks

A GPT-style language model trained on plain text, with a Plotly Dash dashboard
that visualises everything happening during training and loss curves, attention
patterns, token embeddings, and a live text generation panel.

---

#### Phase 1: Transformer from scratch

We build the model in PyTorch. This covers a
character-level tokenizer, multi-head causal self-attention with masking, a
feed-forward network, layer normalisation, and the full autoregressive training
loop with AdamW and cosine learning rate decay.


---

#### Phase 2: Fine-tuning GPT-2 with LoRA

Instead of training from random weights, we take a pre-trained GPT-2 and
adapt it to text using LoRA (Low-Rank Adaptation). Only a small
fraction of the parameters are updated, which means we can run this on a
laptop. 

---

#### Phase 3:  ML Dashboard

A Dash app reads the training logs written by Phase 1 and 2 and renders them
live as the model trains. Six tabs cover the loss and perplexity curves,
the learning rate schedule, per-layer attention heatmaps, a PCA or t-SNE
scatter of token embeddings, and an interactive generation panel where you
can type a prompt and adjust temperature, top-k, top-p, and beam width in
real time.

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

