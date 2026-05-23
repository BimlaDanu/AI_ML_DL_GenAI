"""
dashboard/app.py
================
Main Plotly Dash entrypoint for the LLM training dashboard.

Tabs:
  1. Overview      — 2×2 subplot: loss, perplexity, LR, grad norm
  2. Loss          — detailed loss + val loss chart
  3. Perplexity    — perplexity curves
  4. Attention     — attention heatmap with layer/head selectors
  5. Embeddings    — PCA / t-SNE token embedding scatter
  6. Inference     — interactive text generation panel

The dashboard auto-refreshes training charts every 10 seconds so you can
watch training progress live while train_scratch.py runs in another terminal.

Run:
    cd llm_project
    python -m dashboard.app

Then open http://127.0.0.1:8050 in browser.

Optional: to connect to a trained model for the Inference tab, set
  MODEL_CHECKPOINT and TOKENIZER_PATH at the bottom of this file.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

import dash
from dash import dcc, html, Input, Output, callback

from dashboard.charts     import (
    build_loss_chart, build_perplexity_chart, build_lr_chart,
    build_grad_norm_chart, build_attention_heatmap,
    build_embedding_scatter, build_training_overview,
)
from dashboard.inference_ui import inference_layout, register_callbacks



# PATHS  ← edit these to match your output files
LOG_CSV       = "training_log.csv"       # written by train_scratch.py
ATTN_NPY      = "attn_weights.npy"       # (n_layers, n_heads, T, T)
EMBED_NPY     = "embeddings.npy"         # (vocab_size, d_model)
MODEL_CKPT    = "checkpoints/best.pt"    # set to None to disable Inference tab
TOKENIZER_PKL = "checkpoints/tokenizer.pkl"

REFRESH_MS    = 10_000    # auto-refresh interval for training charts (milliseconds)



# THEME
COLORS = {
    "bg":     "#0f1117",
    "card":   "#1a1d27",
    "border": "#2a2d3a",
    "text":   "#e0e0e0",
    "muted":  "#888888",
    "accent": "#7c6af7",
    "green":  "#34d399",
}

_tab_style = {
    "backgroundColor": COLORS["card"],
    "color":           COLORS["muted"],
    "border":          f"1px solid {COLORS['border']}",
    "padding":         "10px 18px",
    "fontFamily":      "Inter, sans-serif",
    "fontSize":        "13px",
}
_tab_selected_style = {
    **_tab_style,
    "backgroundColor": COLORS["accent"],
    "color":           "#fff",
    "fontWeight":      "600",
    "borderColor":     COLORS["accent"],
}
_card = {
    "backgroundColor": COLORS["card"],
    "borderRadius":    "10px",
    "padding":         "20px",
    "marginBottom":    "16px",
    "border":          f"1px solid {COLORS['border']}",
}
_graph_config = {"displayModeBar": True, "displaylogo": False}


# HELPERS
def load_log() -> pd.DataFrame:
    """Load training CSV; return empty DataFrame if file not found."""
    if not os.path.exists(LOG_CSV):
        return pd.DataFrame(columns=[
            "step", "train_loss", "val_loss",
            "train_perplexity", "val_perplexity",
            "lr", "grad_norm", "elapsed_sec", "epoch",
        ])
    df = pd.read_csv(LOG_CSV)
    for col in ["train_loss", "val_loss", "train_perplexity", "val_perplexity",
                "lr", "grad_norm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_attn() -> np.ndarray:
    """Load attention weights; return zeros if file not found."""
    if os.path.exists(ATTN_NPY):
        return np.load(ATTN_NPY)
    return np.zeros((4, 4, 32, 32))   # dummy shape


def load_embeddings() -> np.ndarray:
    """Load token embeddings; return zeros if file not found."""
    if os.path.exists(EMBED_NPY):
        return np.load(EMBED_NPY)
    return np.zeros((65, 128))         # dummy shape


def load_model_and_tokenizer():
    """
    Load the trained GPT model and tokenizer for the Inference tab.
    Returns (None, None) if checkpoint or tokenizer is missing.
    """
    if not MODEL_CKPT or not os.path.exists(MODEL_CKPT):
        return None, None
    if not os.path.exists(TOKENIZER_PKL):
        return None, None

    from data.prepare      import CharTokenizer
    from model.transformer import GPT, GPTConfig

    tokenizer = CharTokenizer().load(TOKENIZER_PKL)

    ckpt      = torch.load(MODEL_CKPT, map_location="cpu")
    cfg_dict  = ckpt.get("config", {})

    model_cfg = GPTConfig(
        vocab_size  = tokenizer.vocab_size,
        context_len = cfg_dict.get("context_len", 128),
        n_layers    = cfg_dict.get("n_layers",    4),
        n_heads     = cfg_dict.get("n_heads",     4),
        d_model     = cfg_dict.get("d_model",     128),
        d_ff        = cfg_dict.get("d_ff",        512),
    )
    model = GPT(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[Dashboard] Model loaded from {MODEL_CKPT}")
    return model, tokenizer


# APP INIT
app = dash.Dash(
    __name__,
    title       = "LLM Training Dashboard",
    update_title= None,
    meta_tags   = [{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.config.suppress_callback_exceptions = True

model, tokenizer = load_model_and_tokenizer()
inference_enabled = model is not None

device = "cuda" if torch.cuda.is_available() else "cpu"



# LAYOUT
def make_header() -> html.Div:
    status_color = COLORS["green"] if inference_enabled else COLORS["muted"]
    status_text  = "Model loaded ✓" if inference_enabled else "No model checkpoint"
    return html.Div([
        html.Div([
            html.H1("🧠 LLM Training Dashboard",
                    style={"color": COLORS["text"], "fontSize": "22px",
                           "fontWeight": "700", "margin": "0",
                           "fontFamily": "Inter, sans-serif"}),
            html.Span(status_text,
                      style={"color": status_color, "fontSize": "12px",
                             "marginLeft": "16px", "fontFamily": "monospace"}),
        ], style={"display": "flex", "alignItems": "center"}),
        html.P(
            f"Auto-refreshes every {REFRESH_MS//1000}s · {LOG_CSV}",
            style={"color": COLORS["muted"], "fontSize": "11px",
                   "margin": "4px 0 0 0", "fontFamily": "monospace"},
        ),
    ], style={
        "backgroundColor": COLORS["card"],
        "padding": "16px 24px",
        "borderBottom": f"1px solid {COLORS['border']}",
        "marginBottom": "0",
    })


def make_stats_bar(df: pd.DataFrame) -> html.Div:
    """Top-level metric cards: latest step, loss, val_ppl, elapsed."""
    def stat(label, value, color=COLORS["text"]):
        return html.Div([
            html.P(label, style={"color": COLORS["muted"], "fontSize": "11px",
                                 "margin": "0", "textTransform": "uppercase",
                                 "letterSpacing": "0.07em"}),
            html.P(value, style={"color": color, "fontSize": "22px",
                                 "fontWeight": "700", "margin": "4px 0 0 0",
                                 "fontFamily": "monospace"}),
        ], style={**_card, "minWidth": "130px", "textAlign": "center",
                  "marginRight": "12px", "marginBottom": "0"})

    if df.empty:
        return html.Div([stat("Step", "—"), stat("Train loss", "—"),
                         stat("Val ppl", "—"), stat("Elapsed", "—")],
                        style={"display": "flex", "padding": "16px 24px"})

    last      = df.iloc[-1]
    val_row   = df.dropna(subset=["val_perplexity"])
    val_ppl   = f"{val_row.iloc[-1]['val_perplexity']:.2f}" if not val_row.empty else "—"
    elapsed   = f"{last.get('elapsed_sec', 0)/60:.1f} min"
    train_loss= f"{last.get('train_loss', float('nan')):.4f}" \
                if not pd.isna(last.get("train_loss", float("nan"))) else "—"

    return html.Div([
        stat("Step",       str(int(last["step"])),   COLORS["accent"]),
        stat("Train loss", train_loss,               COLORS["accent"]),
        stat("Val ppl",    val_ppl,                  COLORS["green"]),
        stat("Elapsed",    elapsed,                  COLORS["text"]),
    ], style={"display": "flex", "padding": "16px 24px",
              "borderBottom": f"1px solid {COLORS['border']}"})


# ── Tab content helpers ───────────────────────────────────────────────────────
def _graph(fig_id, height="420px"):
    return dcc.Graph(id=fig_id, config=_graph_config, style={"height": height})


def overview_tab():
    return html.Div([_graph("fig-overview", "600px")],
                    style={"padding": "20px"})


def loss_tab():
    return html.Div([
        html.Div([_graph("fig-loss")],   style=_card),
        html.Div([_graph("fig-lr")],     style=_card),
        html.Div([_graph("fig-gnorm")],  style=_card),
    ], style={"padding": "20px"})


def perplexity_tab():
    return html.Div([
        html.Div([_graph("fig-ppl")], style=_card),
    ], style={"padding": "20px"})


def attention_tab(n_layers=4, n_heads=4):
    return html.Div([
        html.Div([
            html.Div([
                html.P("LAYER", style={"color": COLORS["muted"], "fontSize": "11px",
                                       "textTransform": "uppercase", "marginBottom": "6px"}),
                dcc.Slider(id="attn-layer", min=0, max=n_layers-1, step=1, value=0,
                           marks={i: str(i) for i in range(n_layers)},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], style={"width": "45%", "marginRight": "40px"}),
            html.Div([
                html.P("HEAD", style={"color": COLORS["muted"], "fontSize": "11px",
                                      "textTransform": "uppercase", "marginBottom": "6px"}),
                dcc.Slider(id="attn-head", min=0, max=n_heads-1, step=1, value=0,
                           marks={i: str(i) for i in range(n_heads)},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], style={"width": "45%"}),
        ], style={**_card, "display": "flex", "alignItems": "center"}),
        html.Div([_graph("fig-attn", "480px")], style=_card),
    ], style={"padding": "20px"})


def embedding_tab():
    return html.Div([
        html.Div([
            html.P("REDUCTION METHOD", style={"color": COLORS["muted"], "fontSize": "11px",
                                              "textTransform": "uppercase", "marginBottom": "8px"}),
            dcc.RadioItems(
                id="emb-method",
                options=[{"label": " PCA", "value": "pca"},
                         {"label": " t-SNE", "value": "tsne"}],
                value="pca",
                inline=True,
                style={"color": COLORS["text"], "fontSize": "13px"},
                inputStyle={"marginRight": "5px", "accentColor": COLORS["accent"]},
                labelStyle={"marginRight": "20px"},
            ),
        ], style=_card),
        html.Div([_graph("fig-emb", "520px")], style=_card),
    ], style={"padding": "20px"})


# ── Full app layout ───────────────────────────────────────────────────────────
tabs = [
    dcc.Tab(label="Overview",    value="tab-overview",    style=_tab_style, selected_style=_tab_selected_style),
    dcc.Tab(label="Loss",        value="tab-loss",        style=_tab_style, selected_style=_tab_selected_style),
    dcc.Tab(label="Perplexity",  value="tab-ppl",         style=_tab_style, selected_style=_tab_selected_style),
    dcc.Tab(label="Attention",   value="tab-attn",        style=_tab_style, selected_style=_tab_selected_style),
    dcc.Tab(label="Embeddings",  value="tab-emb",         style=_tab_style, selected_style=_tab_selected_style),
]
if inference_enabled:
    tabs.append(dcc.Tab(label="🔮 Inference", value="tab-inf",
                        style=_tab_style, selected_style=_tab_selected_style))

app.layout = html.Div([
    make_header(),
    html.Div(id="stats-bar"),
    dcc.Tabs(id="tabs", value="tab-overview", children=tabs,
             style={"borderBottom": f"1px solid {COLORS['border']}",
                    "backgroundColor": COLORS["card"]}),
    html.Div(id="tab-content"),
    # Auto-refresh interval
    dcc.Interval(id="refresh-interval", interval=REFRESH_MS, n_intervals=0),
], style={"backgroundColor": COLORS["bg"], "minHeight": "100vh",
          "fontFamily": "Inter, sans-serif"})



# CALLBACKS
@app.callback(Output("stats-bar",   "children"),
              Output("tab-content", "children"),
              Input("tabs",                 "value"),
              Input("refresh-interval",     "n_intervals"))
def render_tab(tab, _n):
    df = load_log()

    # Detect model config from checkpoint for slider maxes
    attn  = load_attn()
    n_layers, n_heads = attn.shape[0], attn.shape[1]

    stats   = make_stats_bar(df)

    if tab == "tab-overview":
        content = overview_tab()
    elif tab == "tab-loss":
        content = loss_tab()
    elif tab == "tab-ppl":
        content = perplexity_tab()
    elif tab == "tab-attn":
        content = attention_tab(n_layers, n_heads)
    elif tab == "tab-emb":
        content = embedding_tab()
    elif tab == "tab-inf" and inference_enabled:
        content = inference_layout()
    else:
        content = html.P("Select a tab above.",
                         style={"color": COLORS["muted"], "padding": "40px"})

    return stats, content


# ── Chart update callbacks 
@app.callback(Output("fig-overview", "figure"),
              Input("refresh-interval", "n_intervals"),
              Input("tabs", "value"))
def update_overview(_, tab):
    if tab != "tab-overview":
        return dash.no_update
    return build_training_overview(load_log())


@app.callback(Output("fig-loss",  "figure"),
              Output("fig-lr",    "figure"),
              Output("fig-gnorm", "figure"),
              Input("refresh-interval", "n_intervals"),
              Input("tabs", "value"))
def update_loss_tab(_, tab):
    if tab != "tab-loss":
        return dash.no_update, dash.no_update, dash.no_update
    df = load_log()
    return build_loss_chart(df), build_lr_chart(df), build_grad_norm_chart(df)


@app.callback(Output("fig-ppl", "figure"),
              Input("refresh-interval", "n_intervals"),
              Input("tabs", "value"))
def update_ppl_tab(_, tab):
    if tab != "tab-ppl":
        return dash.no_update
    return build_perplexity_chart(load_log())


@app.callback(Output("fig-attn", "figure"),
              Input("attn-layer",        "value"),
              Input("attn-head",         "value"),
              Input("refresh-interval",  "n_intervals"))
def update_attn(layer, head, _):
    weights = load_attn()
    return build_attention_heatmap(weights, layer=layer or 0, head=head or 0)


@app.callback(Output("fig-emb", "figure"),
              Input("emb-method",        "value"),
              Input("refresh-interval",  "n_intervals"))
def update_emb(method, _):
    emb    = load_embeddings()
    # Try to load real token labels from the tokenizer
    labels = [str(i) for i in range(emb.shape[0])]
    if os.path.exists(TOKENIZER_PKL):
        try:
            from data.prepare import CharTokenizer
            tok    = CharTokenizer().load(TOKENIZER_PKL)
            labels = [tok.idx2char.get(i, str(i)) for i in range(emb.shape[0])]
        except Exception:
            pass
    return build_embedding_scatter(emb, labels, method=method or "pca")


# ── Register inference callbacks if model is loaded ──────────────────────────
if inference_enabled:
    register_callbacks(app, model, tokenizer, device=device)


# ENTRY POINT
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  LLM Training Dashboard")
    print(f"  Open → http://127.0.0.1:8050")
    print(f"  Log  → {os.path.abspath(LOG_CSV)}")
    print(f"  Inference {'enabled ✓' if inference_enabled else 'disabled (no checkpoint)'}")
    print(f"{'='*60}\n")
    app.run(debug=True, host="0.0.0.0", port=8050)

