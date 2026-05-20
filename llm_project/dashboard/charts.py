"""
dashboard/charts.py
===================
Pure Plotly figure builders — no Dash here, just functions that return
go.Figure objects. The Dash app imports these and drops them into dcc.Graph.

Functions:
  build_loss_chart(df)           — train + val loss over steps
  build_perplexity_chart(df)     — train + val perplexity over steps
  build_lr_chart(df)             — learning rate schedule
  build_grad_norm_chart(df)      — gradient norm over steps
  build_attention_heatmap(weights, layer, head)  — (T×T) attention matrix
  build_embedding_scatter(emb, labels, method)   — PCA or t-SNE 2D scatter
  build_token_prob_bar(token_probs, inv_vocab)   — top-k next-token bar chart

All figures use a consistent dark theme matching a typical ML dashboard.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional


# THEME
THEME = dict(
    bg       = "#0f1117",
    paper_bg = "#1a1d27",
    grid     = "#2a2d3a",
    text     = "#e0e0e0",
    accent1  = "#7c6af7",   # purple — train metrics
    accent2  = "#34d399",   # green  — val metrics
    accent3  = "#f59e0b",   # amber  — lr / aux
    accent4  = "#f87171",   # red    — warnings / grad norm
)

BASE_LAYOUT = dict(
    paper_bgcolor = THEME["paper_bg"],
    plot_bgcolor  = THEME["bg"],
    font          = dict(color=THEME["text"], family="Inter, sans-serif", size=12),
    margin        = dict(l=50, r=20, t=50, b=40),
    xaxis         = dict(gridcolor=THEME["grid"], zerolinecolor=THEME["grid"]),
    yaxis         = dict(gridcolor=THEME["grid"], zerolinecolor=THEME["grid"]),
    legend        = dict(bgcolor=THEME["paper_bg"], bordercolor=THEME["grid"]),
    hovermode     = "x unified",
)


def _apply_theme(fig: go.Figure, title: str = "") -> go.Figure:
    layout = dict(BASE_LAYOUT)
    layout["title"] = dict(text=title, font=dict(size=15, color=THEME["text"]))
    fig.update_layout(**layout)
    return fig



# 1. LOSS CHART
def build_loss_chart(df: pd.DataFrame) -> go.Figure:
    """
    Line chart of training loss and validation loss over training steps.

    Expects df columns: step, train_loss, val_loss (val_loss may have NaN gaps).

    The train line updates every log_interval steps.
    The val line updates every eval_interval steps (sparser).
    """
    fig = go.Figure()

    # Train loss — every row has this
    train_df = df.dropna(subset=["train_loss"])
    fig.add_trace(go.Scatter(
        x    = train_df["step"],
        y    = train_df["train_loss"],
        name = "Train loss",
        mode = "lines",
        line = dict(color=THEME["accent1"], width=2),
        hovertemplate="step %{x}<br>train_loss: %{y:.4f}<extra></extra>",
    ))

    # Validation loss — sparser, only eval rows
    val_df = df.dropna(subset=["val_loss"])
    if not val_df.empty:
        fig.add_trace(go.Scatter(
            x    = val_df["step"],
            y    = val_df["val_loss"],
            name = "Val loss",
            mode = "lines+markers",
            line = dict(color=THEME["accent2"], width=2, dash="dot"),
            marker= dict(size=6, color=THEME["accent2"]),
            hovertemplate="step %{x}<br>val_loss: %{y:.4f}<extra></extra>",
        ))

    fig.update_layout(
        xaxis_title="Step",
        yaxis_title="Cross-entropy loss",
    )
    return _apply_theme(fig, "Training & Validation Loss")



# 2. PERPLEXITY CHART
def build_perplexity_chart(df: pd.DataFrame) -> go.Figure:
    """
    Line chart of perplexity (exp(loss)) over steps.

    Perplexity intuition:
      ppl=1   → perfect prediction
      ppl=10  → as confused as picking from 10 equally likely options
      Lower is better; expect dramatic drops in the first few hundred steps.
    """
    fig = go.Figure()

    train_df = df.dropna(subset=["train_perplexity"])
    fig.add_trace(go.Scatter(
        x    = train_df["step"],
        y    = train_df["train_perplexity"],
        name = "Train ppl",
        mode = "lines",
        line = dict(color=THEME["accent1"], width=2),
        hovertemplate="step %{x}<br>train_ppl: %{y:.2f}<extra></extra>",
    ))

    val_df = df.dropna(subset=["val_perplexity"])
    if not val_df.empty:
        fig.add_trace(go.Scatter(
            x    = val_df["step"],
            y    = val_df["val_perplexity"],
            name = "Val ppl",
            mode = "lines+markers",
            line = dict(color=THEME["accent2"], width=2, dash="dot"),
            marker= dict(size=6),
            hovertemplate="step %{x}<br>val_ppl: %{y:.2f}<extra></extra>",
        ))

    fig.update_layout(xaxis_title="Step", yaxis_title="Perplexity")
    return _apply_theme(fig, "Perplexity")



# 3. LEARNING RATE CHART
def build_lr_chart(df: pd.DataFrame) -> go.Figure:
    """
    Area chart of the learning rate schedule (warmup + cosine decay).
    """
    fig = go.Figure()

    lr_df = df.dropna(subset=["lr"])
    fig.add_trace(go.Scatter(
        x    = lr_df["step"],
        y    = lr_df["lr"],
        name = "Learning rate",
        mode = "lines",
        fill = "tozeroy",
        line = dict(color=THEME["accent3"], width=2),
        fillcolor = "rgba(245, 158, 11, 0.1)",
        hovertemplate="step %{x}<br>lr: %{y:.2e}<extra></extra>",
    ))

    fig.update_layout(
        xaxis_title="Step",
        yaxis_title="Learning rate",
        yaxis=dict(tickformat=".1e", gridcolor=THEME["grid"]),
    )
    return _apply_theme(fig, "Learning Rate Schedule")


# 4. GRADIENT NORM CHART
def build_grad_norm_chart(df: pd.DataFrame) -> go.Figure:
    """
    Gradient norm over steps. Spikes indicate instability.
    The horizontal line shows the clip value (1.0 by default).
    """
    fig = go.Figure()

    gn_df = df.dropna(subset=["grad_norm"])
    fig.add_trace(go.Scatter(
        x    = gn_df["step"],
        y    = gn_df["grad_norm"],
        name = "Grad norm",
        mode = "lines",
        line = dict(color=THEME["accent4"], width=1.5),
        hovertemplate="step %{x}<br>grad_norm: %{y:.4f}<extra></extra>",
    ))

    # Clip line
    if not gn_df.empty:
        fig.add_hline(
            y=1.0, line_dash="dash", line_color=THEME["grid"],
            annotation_text="clip=1.0",
            annotation_font_color=THEME["text"],
        )

    fig.update_layout(xaxis_title="Step", yaxis_title="Gradient norm")
    return _apply_theme(fig, "Gradient Norm")

# 5. ATTENTION HEATMAP
def build_attention_heatmap(
    weights: np.ndarray,    # (n_layers, n_heads, T, T)
    layer: int = 0,
    head: int  = 0,
    token_labels: Optional[list[str]] = None,
) -> go.Figure:
    """
    Heatmap of the attention weight matrix for a given layer and head.

    Each cell [i, j] shows how much token i attends to token j.
    The causal mask means the upper triangle is always zero.

    Parameters
    ----------
    weights      : (n_layers, n_heads, T, T) array from attn_weights.npy
    layer        : which layer to visualise (0-indexed)
    head         : which attention head to visualise (0-indexed)
    token_labels : optional list of T token strings for axis tick labels
    """
    n_layers, n_heads, T, _ = weights.shape
    layer = min(layer, n_layers - 1)
    head  = min(head,  n_heads - 1)

    matrix = weights[layer, head]   # (T, T)

    ticks = token_labels if token_labels else [str(i) for i in range(T)]
    # Keep labels short for readability
    if len(ticks) > 20:
        ticks = [t if i % max(1, T // 20) == 0 else "" for i, t in enumerate(ticks)]

    fig = go.Figure(go.Heatmap(
        z           = matrix,
        x           = ticks,
        y           = ticks,
        colorscale  = "Blues",
        reversescale= False,
        showscale   = True,
        hovertemplate="from token %{y} → to token %{x}<br>weight: %{z:.4f}<extra></extra>",
    ))

    fig.update_layout(
        xaxis_title = "Key (attended to)",
        yaxis_title = "Query (attending from)",
        yaxis       = dict(autorange="reversed", gridcolor=THEME["grid"]),
        xaxis       = dict(gridcolor=THEME["grid"]),
    )
    return _apply_theme(fig, f"Attention Weights — Layer {layer}, Head {head}")


# 6. EMBEDDING SCATTER (PCA / t-SNE)
def build_embedding_scatter(
    emb: np.ndarray,               # (vocab_size, d_model)
    labels: list[str],             # one string per token
    method: str = "pca",           # "pca" or "tsne"
    perplexity: int = 30,          # t-SNE perplexity (ignored for PCA)
) -> go.Figure:
    """
    2D scatter plot of token embeddings reduced with PCA or t-SNE.

    Each point represents one vocabulary token.
    Tokens with similar meanings should cluster together once the model
    has learned meaningful representations.

    Parameters
    ----------
    emb        : (vocab_size, d_model) embedding matrix
    labels     : token strings (from tokenizer.idx2char.values())
    method     : "pca" or "tsne"
    perplexity : t-SNE perplexity (smaller vocab → smaller value)
    """
    from sklearn.decomposition import PCA

    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
            reducer = TSNE(
                n_components=2,
                perplexity=min(perplexity, max(5, len(labels) - 1)),
                random_state=42,
                max_iter=1000,
            )
            coords = reducer.fit_transform(emb)
            method_label = "t-SNE"
        except Exception as e:
            print(f"[Warning] t-SNE failed ({e}), falling back to PCA")
            method = "pca"

    if method == "pca":
        coords = PCA(n_components=2, random_state=42).fit_transform(emb)
        method_label = "PCA"

    fig = go.Figure(go.Scatter(
        x    = coords[:, 0],
        y    = coords[:, 1],
        mode = "markers+text",
        text = labels,
        textposition = "top center",
        textfont     = dict(size=9, color=THEME["text"]),
        marker       = dict(
            size   = 8,
            color  = list(range(len(labels))),
            colorscale = "Viridis",
            showscale  = False,
            opacity    = 0.85,
        ),
        hovertemplate="token: %{text}<br>x: %{x:.3f}  y: %{y:.3f}<extra></extra>",
    ))

    fig.update_layout(
        xaxis_title = f"{method_label} component 1",
        yaxis_title = f"{method_label} component 2",
    )
    return _apply_theme(fig, f"Token Embeddings ({method_label})")


# 7. TOKEN PROBABILITY BAR CHART
def build_token_prob_bar(
    token_probs: list[tuple[int, float]],   # [(token_id, probability), ...]
    inv_vocab: dict[int, str],              # idx2char from tokenizer
) -> go.Figure:
    """
    Horizontal bar chart showing the top-k next-token probabilities.

    Use this in the Dash inference UI to show what the model is "thinking"
    at each generation step.

    Parameters
    ----------
    token_probs : list of (token_id, probability) sorted descending
    inv_vocab   : mapping from token_id → token string
    """
    token_strs = [repr(inv_vocab.get(tid, "?")) for tid, _ in token_probs]
    probs      = [prob for _, prob in token_probs]

    fig = go.Figure(go.Bar(
        x           = probs,
        y           = token_strs,
        orientation = "h",
        marker_color = [
            THEME["accent1"] if i == 0 else THEME["accent3"]
            for i in range(len(probs))
        ],
        hovertemplate="token: %{y}<br>probability: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        xaxis_title = "Probability",
        yaxis       = dict(autorange="reversed", gridcolor=THEME["grid"]),
        xaxis       = dict(range=[0, 1], gridcolor=THEME["grid"]),
        showlegend  = False,
        bargap      = 0.15,
    )
    return _apply_theme(fig, "Next-Token Probabilities")


# 8. COMBINED TRAINING OVERVIEW  (2×2 subplot)
def build_training_overview(df: pd.DataFrame) -> go.Figure:
    """
    2×2 subplot overview: loss | perplexity | lr | grad_norm.
    Useful as the landing page of the dashboard.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Loss", "Perplexity", "Learning Rate", "Gradient Norm"],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    def add(trace, row, col):
        fig.add_trace(trace, row=row, col=col)

    # Loss
    t = df.dropna(subset=["train_loss"])
    add(go.Scatter(x=t["step"], y=t["train_loss"], name="Train loss",
                   line=dict(color=THEME["accent1"], width=2), showlegend=True), 1, 1)
    v = df.dropna(subset=["val_loss"])
    if not v.empty:
        add(go.Scatter(x=v["step"], y=v["val_loss"], name="Val loss",
                       line=dict(color=THEME["accent2"], width=2, dash="dot"),
                       mode="lines+markers", showlegend=True), 1, 1)

    # Perplexity
    t2 = df.dropna(subset=["train_perplexity"])
    add(go.Scatter(x=t2["step"], y=t2["train_perplexity"], name="Train ppl",
                   line=dict(color=THEME["accent1"], width=2), showlegend=False), 1, 2)
    v2 = df.dropna(subset=["val_perplexity"])
    if not v2.empty:
        add(go.Scatter(x=v2["step"], y=v2["val_perplexity"], name="Val ppl",
                       line=dict(color=THEME["accent2"], width=2, dash="dot"),
                       mode="lines+markers", showlegend=False), 1, 2)

    # LR
    lr = df.dropna(subset=["lr"])
    add(go.Scatter(x=lr["step"], y=lr["lr"], name="LR",
                   line=dict(color=THEME["accent3"], width=2),
                   fill="tozeroy", fillcolor="rgba(245,158,11,0.1)",
                   showlegend=False), 2, 1)

    # Grad norm
    gn = df.dropna(subset=["grad_norm"])
    add(go.Scatter(x=gn["step"], y=gn["grad_norm"], name="Grad norm",
                   line=dict(color=THEME["accent4"], width=1.5),
                   showlegend=False), 2, 2)

    fig.update_layout(
        paper_bgcolor = THEME["paper_bg"],
        plot_bgcolor  = THEME["bg"],
        font          = dict(color=THEME["text"], family="Inter, sans-serif"),
        margin        = dict(l=50, r=20, t=60, b=40),
        title         = dict(text="Training Overview", font=dict(size=16)),
        legend        = dict(bgcolor=THEME["paper_bg"]),
    )
    fig.update_xaxes(gridcolor=THEME["grid"], zerolinecolor=THEME["grid"])
    fig.update_yaxes(gridcolor=THEME["grid"], zerolinecolor=THEME["grid"])

    return fig

