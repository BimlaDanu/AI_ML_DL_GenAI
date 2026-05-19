### LLM Project Builder

- Building tiny LLM from scratch in three phases.

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

---

#### Phase 1:  Train a tiny GPT from scratch

```bash
cd llm_project
python -m train.train_scratch
```

Outputs written automatically:
| File | Used by |
|---|---|
| `training_log.csv` | Dashboard loss / ppl / LR charts |
| `attn_weights.npy` | Dashboard attention heatmap |
| `embeddings.npy` | Dashboard PCA / t-SNE scatter |
| `checkpoints/best.pt` | Inference tab |
| `checkpoints/tokenizer.pkl` | Inference tab |

---

#### Phase 2: Fine-tune GPT with LoRA

Install extra deps first:

```bash
pip install transformers peft datasets accelerate
```

Then:

```bash
python -m train.finetune_gpt2
```

---

#### Dashboard

Open the dashboard While training starts:

```bash
python -m dashboard.app
```
on web:

http://127.0.0.1:8050


Tabs:
- **Overview:** — 2×2 live training summary
- **Loss:** — train + val loss, LR schedule, grad norm
- **Perplexity:** — perplexity curves
- **Attention:** — per-layer, per-head attention heatmap
- **Embeddings:** — PCA or t-SNE token embedding scatter
- **Inference:** — generate text with temperature / top-k / top-p / beam search sliders

---

#### Optimization

To training on text `train/train_scratch.py`:

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

