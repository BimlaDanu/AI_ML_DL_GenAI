"""
dashboard/inference_ui.py
=========================
Dash layout and callbacks for the interactive text generation panel.

Provides:
  inference_layout()   — returns the Dash layout div for the inference tab
  register_callbacks() — registers all Dash callbacks for the inference UI

The panel lets you:
  - Type a prompt
  - Adjust temperature, top-k, top-p, max_new_tokens via sliders
  - Click Generate → see generated text + top-k token probability bar chart
  - Switch between greedy / sampling / beam search

Usage (in app.py):
    from dashboard.inference_ui import inference_layout, register_callbacks
    register_callbacks(app, model, tokenizer, device)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from dash import dcc, html, Input, Output, State

from dashboard.charts import build_token_prob_bar
from model.generate   import generate, beam_search, get_next_token_probs, GenerationConfig


# THEME  (matches charts.py)
COLORS = {
    "bg":       "#0f1117",
    "card":     "#1a1d27",
    "border":   "#2a2d3a",
    "text":     "#e0e0e0",
    "muted":    "#888",
    "accent":   "#7c6af7",
    "green":    "#34d399",
    "amber":    "#f59e0b",
}

_card = {
    "backgroundColor": COLORS["card"],
    "borderRadius": "10px",
    "padding": "20px",
    "marginBottom": "16px",
    "border": f"1px solid {COLORS['border']}",
}

_label = {
    "color": COLORS["muted"],
    "fontSize": "11px",
    "textTransform": "uppercase",
    "letterSpacing": "0.08em",
    "marginBottom": "6px",
}

_slider_style = {"marginBottom": "22px"}


# LAYOUT
def inference_layout() -> html.Div:
    """
    Return the complete Dash layout for the inference / generation tab.

    Structure:
      Left column  : prompt input + sliders + generate button
      Right column : generated text output + token probability bar chart
    """
    return html.Div([

        # ── Row ──────────────────────────────────────────────────────────────
        html.Div([

            # ── Left: controls ───────────────────────────────────────────────
            html.Div([

                html.Div([
                    html.P("PROMPT", style=_label),
                    dcc.Textarea(
                        id="inf-prompt",
                        value="To be or not to be",
                        placeholder="Enter your prompt here...",
                        style={
                            "width": "100%", "height": "90px",
                            "backgroundColor": COLORS["bg"],
                            "color": COLORS["text"],
                            "border": f"1px solid {COLORS['border']}",
                            "borderRadius": "6px",
                            "padding": "10px",
                            "fontFamily": "monospace",
                            "fontSize": "13px",
                            "resize": "vertical",
                        },
                    ),
                ], style=_card),

                html.Div([
                    html.P("GENERATION STRATEGY", style=_label),
                    dcc.RadioItems(
                        id="inf-strategy",
                        options=[
                            {"label": " Sampling  ", "value": "sample"},
                            {"label": " Greedy",    "value": "greedy"},
                            {"label": " Beam search", "value": "beam"},
                        ],
                        value="sample",
                        inline=True,
                        style={"color": COLORS["text"], "fontSize": "13px"},
                        inputStyle={"marginRight": "5px", "accentColor": COLORS["accent"]},
                        labelStyle={"marginRight": "16px"},
                    ),
                ], style=_card),

                html.Div([

                    html.P("MAX NEW TOKENS", style=_label),
                    dcc.Slider(
                        id="inf-max-tokens", min=10, max=300, step=10, value=100,
                        marks={10: "10", 100: "100", 200: "200", 300: "300"},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),

                    html.P("TEMPERATURE", style={**_label, "marginTop": "18px"}),
                    dcc.Slider(
                        id="inf-temperature", min=0.1, max=2.0, step=0.05, value=0.8,
                        marks={0.1: "0.1", 0.5: "0.5", 1.0: "1.0", 2.0: "2.0"},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),

                    html.P("TOP-K  (0 = disabled)", style={**_label, "marginTop": "18px"}),
                    dcc.Slider(
                        id="inf-top-k", min=0, max=100, step=1, value=40,
                        marks={0: "off", 10: "10", 40: "40", 100: "100"},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),

                    html.P("TOP-P  (1.0 = disabled)", style={**_label, "marginTop": "18px"}),
                    dcc.Slider(
                        id="inf-top-p", min=0.5, max=1.0, step=0.01, value=0.95,
                        marks={0.5: "0.5", 0.9: "0.9", 0.95: "0.95", 1.0: "1.0"},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),

                    html.P("BEAM WIDTH  (ignored unless beam search)", style={**_label, "marginTop": "18px"}),
                    dcc.Slider(
                        id="inf-beam-width", min=1, max=8, step=1, value=4,
                        marks={1: "1", 2: "2", 4: "4", 8: "8"},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),

                ], style=_card),

                html.Button(
                    "▶  Generate",
                    id="inf-generate-btn",
                    n_clicks=0,
                    style={
                        "width": "100%",
                        "padding": "12px",
                        "backgroundColor": COLORS["accent"],
                        "color": "#fff",
                        "border": "none",
                        "borderRadius": "8px",
                        "fontSize": "15px",
                        "fontWeight": "600",
                        "cursor": "pointer",
                        "letterSpacing": "0.04em",
                    },
                ),

                # Loading spinner while generating
                dcc.Loading(
                    id="inf-loading",
                    type="circle",
                    color=COLORS["accent"],
                    children=html.Div(id="inf-loading-output"),
                ),

            ], style={"width": "38%", "paddingRight": "20px"}),

            # ── Right: outputs ───────────────────────────────────────────────
            html.Div([

                html.Div([
                    html.P("GENERATED TEXT", style=_label),
                    html.Div(
                        id="inf-output-text",
                        children="Generated text will appear here...",
                        style={
                            "backgroundColor": COLORS["bg"],
                            "color": COLORS["text"],
                            "fontFamily": "monospace",
                            "fontSize": "13px",
                            "lineHeight": "1.7",
                            "padding": "14px",
                            "borderRadius": "6px",
                            "border": f"1px solid {COLORS['border']}",
                            "minHeight": "140px",
                            "whiteSpace": "pre-wrap",
                            "wordBreak": "break-word",
                        },
                    ),
                ], style=_card),

                html.Div([
                    html.P("NEXT-TOKEN PROBABILITIES  (top-10 after prompt)", style=_label),
                    dcc.Graph(
                        id="inf-token-prob-chart",
                        config={"displayModeBar": False},
                        style={"height": "320px"},
                    ),
                ], style=_card),

            ], style={"width": "62%"}),

        ], style={"display": "flex", "alignItems": "flex-start"}),

    ], style={"backgroundColor": COLORS["bg"], "padding": "20px"})



# CALLBACKS
def register_callbacks(app, model, tokenizer, device: str = "cpu") -> None:
    """
    Register Dash callbacks that connect the inference UI to the model.

    Parameters
    ----------
    app       : Dash app instance
    model     : trained GPT instance (from model/transformer.py)
    tokenizer : fitted CharTokenizer (from data/prepare.py)
    device    : 'cpu' or 'cuda'
    """

    @app.callback(
        Output("inf-output-text",       "children"),
        Output("inf-token-prob-chart",  "figure"),
        Output("inf-loading-output",    "children"),
        Input("inf-generate-btn",       "n_clicks"),
        State("inf-prompt",             "value"),
        State("inf-strategy",           "value"),
        State("inf-max-tokens",         "value"),
        State("inf-temperature",        "value"),
        State("inf-top-k",              "value"),
        State("inf-top-p",              "value"),
        State("inf-beam-width",         "value"),
        prevent_initial_call=True,
    )
    def on_generate(
        n_clicks, prompt, strategy,
        max_tokens, temperature, top_k, top_p, beam_width,
    ):
        """Generate text and update the output panel."""
        if not prompt:
            prompt = "To be"

        prompt_ids = torch.tensor(
            [tokenizer.encode(prompt)], dtype=torch.long
        )

        if strategy == "beam":
            out_ids = beam_search(
                model       = model,
                prompt_ids  = prompt_ids,
                max_new_tokens = int(max_tokens),
                beam_width  = int(beam_width),
                device      = device,
            )
        else:
            gen_cfg = GenerationConfig(
                max_new_tokens = int(max_tokens),
                temperature    = float(temperature),
                top_k          = int(top_k) if strategy == "sample" else 1,
                top_p          = float(top_p) if strategy == "sample" else 1.0,
            )
            out_ids = generate(model, prompt_ids, gen_cfg, device=device)

        generated_text = tokenizer.decode(out_ids[0].tolist())

        # Token probability bar chart (always uses sampling logits)
        token_probs = get_next_token_probs(
            model      = model,
            prompt_ids = prompt_ids,
            top_k      = 10,
            temperature= float(temperature),
            device     = device,
        )
        fig = build_token_prob_bar(token_probs, tokenizer.idx2char)

        return generated_text, fig, ""

