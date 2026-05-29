"""
╔══════════════════════════════════════════════════════════╗
║   ML / AI & Data Science Dashboard  —  Python + Dash    ║
║                                                          ║
║  Run:  pip install dash plotly pandas numpy scikit-learn ║
║        python ml_dashboard.py                            ║
║  Open: http://127.0.0.1:8050                             ║
╚══════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_blobs
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import dash
from dash import dcc, html, dash_table, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ------------------------------------------------------------#
#  THEME & COLOURS
# ------------------------------------------------------------#

BG      = "#020817"
CARD_BG = "rgba(15,23,42,0.85)"
BORDER  = "rgba(56,189,248,0.18)"
ACCENT  = "#38bdf8"
PINK    = "#f472b6"
PURPLE  = "#6366f1"
GREEN   = "#4ade80"
MUTED   = "#64748b"
TEXT    = "#f1f5f9"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="monospace", color=TEXT, size=11),
    margin=dict(l=40, r=20, t=30, b=40),
    xaxis=dict(gridcolor="rgba(56,189,248,0.07)", zerolinecolor="rgba(56,189,248,0.1)"),
    yaxis=dict(gridcolor="rgba(56,189,248,0.07)", zerolinecolor="rgba(56,189,248,0.1)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(56,189,248,0.15)", borderwidth=1),
)

def card(children, style=None):
    base = dict(
        background=CARD_BG, border=f"1px solid {BORDER}", borderRadius=16,
        padding="20px 24px", backdropFilter="blur(12px)", marginBottom=20,
    )
    if style:
        base.update(style)
    return html.Div(children, style=base)

def section_label(text):
    return html.P(text, style=dict(
        color=MUTED, fontSize=10, letterSpacing=3, textTransform="uppercase",
        fontFamily="monospace", marginBottom=10, marginTop=0,
    ))

def input_field(id_, placeholder, type_="text"):
    return dcc.Input(
        id=id_, type=type_, placeholder=placeholder,
        style=dict(
            width="100%", background="rgba(15,23,42,0.9)",
            border=f"1px solid {BORDER}", borderRadius=8,
            padding="9px 14px", color=TEXT, fontSize=12,
            fontFamily="monospace", outline="none", boxSizing="border-box",
        ),
    )

# ------------------------------------------------------------#
#  SYNTHETIC DATA GENERATION
# ------------------------------------------------------------#
np.random.seed(42)
# Classification dataset
X, y = make_classification(n_samples=600, n_features=10, n_informative=6,
                           n_redundant=2, random_state=42)
FEATURE_NAMES = [
    "Age", "Income", "Tenure", "Usage", "Region",
    "Device", "Plan", "Visits", "Spend", "Churn_Hist"
]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

#  Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
}
trained, metrics_data = {}, []
for name, mdl in models.items():
    mdl.fit(X_train_s, y_train)
    trained[name] = mdl
    acc = mdl.score(X_test_s, y_test)
    y_prob = mdl.predict_proba(X_test_s)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    metrics_data.append(dict(Model=name, Accuracy=round(acc*100,1),
                             AUC_ROC=round(roc_auc*100,1)))

metrics_df = pd.DataFrame(metrics_data)

#  Feature importance from RF
rf = trained["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=FEATURE_NAMES).sort_values()

# Simulated training loss
epochs = list(range(1, 21))
train_loss = [0.95 * np.exp(-0.18 * e) + np.random.normal(0, 0.008) for e in epochs]
val_loss   = [0.98 * np.exp(-0.15 * e) + np.random.normal(0, 0.012) + 0.03 for e in epochs]
train_acc  = [1 - tl + np.random.normal(0, 0.005) for tl in train_loss]
val_acc    = [1 - vl + np.random.normal(0, 0.007) for vl in val_loss]

# Cluster data 
X_blob, y_blob = make_blobs(n_samples=300, centers=3, cluster_std=1.4, random_state=7)

# ------------------------------------------------------------#
#  PLOTTING FIGURES
# ------------------------------------------------------------#
def fig_loss_curves():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode="lines+markers",
        name="Train Loss", line=dict(color=ACCENT, width=2.5),
        marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode="lines+markers",
        name="Val Loss", line=dict(color=PINK, width=2.5, dash="dash"),
        marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=epochs, y=train_acc, mode="lines",
        name="Train Acc", line=dict(color=GREEN, width=1.5), visible="legendonly"))
    fig.update_layout(**PLOTLY_LAYOUT, title="Training & Validation Loss",
                      xaxis_title="Epoch", yaxis_title="Loss")
    return fig

def fig_model_compare():
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Accuracy %", x=metrics_df["Model"],
        y=metrics_df["Accuracy"], marker_color=ACCENT, marker_line_width=0))
    fig.add_trace(go.Bar(name="AUC-ROC %", x=metrics_df["Model"],
        y=metrics_df["AUC_ROC"], marker_color=PURPLE, marker_line_width=0))
    fig.update_layout(**PLOTLY_LAYOUT, title="Model Comparison", barmode="group")
    fig.update_yaxes(range=[60, 100], gridcolor="rgba(56,189,248,0.07)")
    #fig.update_layout(**PLOTLY_LAYOUT, title="Model Comparison",
    #                  barmode="group", yaxis=dict(range=[60, 100],
    #                  gridcolor="rgba(56,189,248,0.07)"))
    return fig

def fig_roc_curves():
    fig = go.Figure()
    colors = [ACCENT, PINK, GREEN]
    for (name, mdl), color in zip(trained.items(), colors):
        y_prob = mdl.predict_proba(X_test_s)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
            name=f"{name} (AUC={roc_auc:.2f})", line=dict(color=color, width=2)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
        name="Random", line=dict(color=MUTED, dash="dot", width=1)))
    fig.update_layout(**PLOTLY_LAYOUT, title="ROC Curves",
                      xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate")
    return fig

def fig_feature_importance():
    fig = go.Figure(go.Bar(
        x=importances.values, y=importances.index, orientation="h",
        marker=dict(color=importances.values, colorscale=[[0, PURPLE],[1, ACCENT]],
                    showscale=False),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title="Feature Importance (Random Forest)",
                      xaxis_title="Importance Score")
    return fig

def fig_clusters():
    colors = [ACCENT, PINK, GREEN]
    fig = go.Figure()
    for cluster_id in sorted(set(y_blob)):
        mask = y_blob == cluster_id
        fig.add_trace(go.Scatter(
            x=X_blob[mask, 0], y=X_blob[mask, 1], mode="markers",
            name=f"Cluster {cluster_id}",
            marker=dict(color=colors[cluster_id], size=7, opacity=0.75,
                        line=dict(width=0.5, color="rgba(255,255,255,0.2)")),
        ))
    fig.update_layout(**PLOTLY_LAYOUT, title="K-Means Cluster Visualization",
                      xaxis_title="PC-1", yaxis_title="PC-2")
    return fig

def fig_confusion(model_name):
    mdl = trained[model_name]
    y_pred = mdl.predict(X_test_s)
    cm = confusion_matrix(y_test, y_pred)
    fig = go.Figure(go.Heatmap(
        z=cm, x=["Pred 0","Pred 1"], y=["True 0","True 1"],
        colorscale=[[0,"#020817"],[1,ACCENT]],
        text=cm, texttemplate="%{text}", showscale=False,
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title=f"Confusion Matrix — {model_name}")
    return fig

def fig_skill_radar():
    skills = ["Python", "ML Ops", "Deep Learning", "Statistics", "SQL", "Data Viz", "NLP"]
    scores = [95, 82, 88, 85, 90, 78, 72]
    fig = go.Figure(go.Scatterpolar(
        r=scores + [scores[0]], theta=skills + [skills[0]],
        fill="toself", line_color=ACCENT,
        fillcolor="rgba(56,189,248,0.12)",
        marker=dict(color=ACCENT, size=6),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title="Skill Profile",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100],
                            gridcolor="rgba(56,189,248,0.1)",
                            tickfont=dict(color=MUTED, size=9)),
            angularaxis=dict(gridcolor="rgba(56,189,248,0.1)",
                             tickfont=dict(color=TEXT, size=10)),
        ))
    return fig

# ------------------------------------------------------------#
#  DASHBOARD LAYOUT
# ------------------------------------------------------------#

ROLES = ["ML Engineer", "Data Scientist", "AI Research Engineer",
         "MLOps Engineer", "NLP Engineer"]

#app = dash.Dash(__name__, title="ML/AI Dashboard")
app = dash.Dash(__name__, title="ML/AI Dashboard", suppress_callback_exceptions=True)
app.layout = html.Div(style=dict(
    minHeight="100vh", background=BG,
    fontFamily="'Courier New', monospace", color=TEXT,
    backgroundImage=(
        "radial-gradient(ellipse at 15% 10%, rgba(14,165,233,0.07) 0%, transparent 55%),"
        "radial-gradient(ellipse at 85% 85%, rgba(99,102,241,0.07) 0%, transparent 55%)"
    ),
), children=[

    #  HEADER 
    html.Div(style=dict(
        borderBottom=f"1px solid {BORDER}", padding="22px 40px",
        display="flex", alignItems="center", justifyContent="space-between",
    ), children=[
        html.Div([
            html.Span("● ", style=dict(color=ACCENT, fontSize=10)),
            html.Span("ML · AI · DATA SCIENCE", style=dict(
                color=ACCENT, fontSize=10, letterSpacing=4)),
            html.H1("Intelligence Dashboard", style=dict(
                margin="6px 0 0", fontSize=24, fontWeight=700,
                color=TEXT, letterSpacing=1)),
        ]),
        html.Div([
            html.Span("Live Experiment Tracker", style=dict(
                color=MUTED, fontSize=11, letterSpacing=2)),
        ]),
    ]),

    # TABS 
    html.Div(style=dict(padding="0 40px"), children=[
        dcc.Tabs(id="tabs", value="overview", style=dict(border="none"),
            parent_style=dict(marginTop=24),
            colors=dict(border=BORDER, primary=ACCENT, background=BG),
            children=[
                dcc.Tab(label="Overview",  value="overview",
                        style=dict(color=MUTED, fontFamily="monospace", fontSize=12, background=BG, border=f"1px solid {BORDER}", borderRadius="8px 8px 0 0"),
                        selected_style=dict(color=ACCENT, fontFamily="monospace", fontSize=12, background=CARD_BG, border=f"1px solid {BORDER}", borderRadius="8px 8px 0 0")),
                dcc.Tab(label="Training",  value="training",
                        style=dict(color=MUTED, fontFamily="monospace", fontSize=12, background=BG, border=f"1px solid {BORDER}", borderRadius="8px 8px 0 0"),
                        selected_style=dict(color=ACCENT, fontFamily="monospace", fontSize=12, background=CARD_BG, border=f"1px solid {BORDER}", borderRadius="8px 8px 0 0")),
                dcc.Tab(label="Features",  value="features",
                        style=dict(color=MUTED, fontFamily="monospace", fontSize=12, background=BG, border=f"1px solid {BORDER}", borderRadius="8px 8px 0 0"),
                        selected_style=dict(color=ACCENT, fontFamily="monospace", fontSize=12, background=CARD_BG, border=f"1px solid {BORDER}", borderRadius="8px 8px 0 0")),
                dcc.Tab(label="Apply",     value="apply",
                        style=dict(color=MUTED, fontFamily="monospace", fontSize=12, background=BG, border=f"1px solid {BORDER}", borderRadius="8px 8px 0 0"),
                        selected_style=dict(color=ACCENT, fontFamily="monospace", fontSize=12, background=CARD_BG, border=f"1px solid {BORDER}", borderRadius="8px 8px 0 0")),
            ]
        ),
        html.Div(id="tab-content", style=dict(paddingTop=24, paddingBottom=60)),
    ]),
])

# ------------------------------------------------------------#
#  TAB CONTENT CALLBACK
# ------------------------------------------------------------#

@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):

    # OVERVIEW 
    if tab == "overview":
        # Metric cards
        kpis = [
            ("Best Accuracy", f"{metrics_df['Accuracy'].max():.1f}%", "+2.1%"),
            ("Training Samples", "450", "+15K rows"),
            ("Features",         "10",  "+4 engineered"),
            ("Models Trained",    "3",   "sklearn"),
        ]
        #
        #
        #kpis = [
        #    ("🎯", "Best Accuracy", f"{metrics_df['Accuracy'].max():.1f}%", "+2.1%"),
        #    ("📦", "Training Samples", "450", "+15K rows"),
        #    ("⚙️",  "Features",         "10",  "+4 engineered"),
        #    ("⚡", "Models Trained",    "3",   "sklearn"),
        #]
        #
        #
        metric_row = html.Div(style=dict(display="grid", gridTemplateColumns="repeat(4,1fr)", gap=16, marginBottom=20), children=[
            card([
                html.P( label, style=dict(color=MUTED, fontSize=11, margin="0 0 8px")),
                #html.P(icon + "  " + label, style=dict(color=MUTED, fontSize=11, margin="0 0 8px")),
                html.H2(value, style=dict(color=TEXT, fontSize=26, margin="0 0 4px", fontWeight=700)),
                html.Span(delta, style=dict(color=ACCENT, fontSize=11)),
            ])
            for label, value, delta in kpis
            #for icon, label, value, delta in kpis
        ])

        charts_row = html.Div(style=dict(display="grid", gridTemplateColumns="1.6fr 1fr", gap=16), children=[
            card(dcc.Graph(figure=fig_model_compare(), config=dict(displayModeBar=False), style=dict(height=280))),
            card(dcc.Graph(figure=fig_skill_radar(),   config=dict(displayModeBar=False), style=dict(height=280))),
        ])

        roc_row = card(dcc.Graph(figure=fig_roc_curves(), config=dict(displayModeBar=False), style=dict(height=300)))

        table = card([
            section_label("Model Metrics Summary"),
            dash_table.DataTable(
                data=metrics_df.to_dict("records"),
                columns=[{"name": c, "id": c} for c in metrics_df.columns],
                style_table=dict(overflowX="auto"),
                style_header=dict(backgroundColor="rgba(56,189,248,0.1)", color=ACCENT,
                                  fontFamily="monospace", fontSize=11, border=f"1px solid {BORDER}"),
                style_cell=dict(backgroundColor="rgba(15,23,42,0.6)", color=TEXT,
                                fontFamily="monospace", fontSize=12,
                                border=f"1px solid {BORDER}", padding="8px 14px"),
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": "rgba(56,189,248,0.03)"}
                ],
            )
        ])

        return html.Div([metric_row, charts_row, html.Div(style=dict(marginTop=16), children=[roc_row, table])])

    # TRAINING 
    elif tab == "training":
        model_selector = card([
            section_label("Select model for confusion matrix"),
            dcc.Dropdown(
                id="model-dd",
                options=[{"label": m, "value": m} for m in trained],
                value="Random Forest",
                style=dict(background="rgba(15,23,42,0.9)", color=TEXT,
                           fontFamily="monospace", fontSize=12),
                className="dark-dd",
            ),
            html.Div(id="conf-matrix"),
        ])

        return html.Div([
            html.Div(style=dict(display="grid", gridTemplateColumns="1fr 1fr", gap=16), children=[
                card(dcc.Graph(figure=fig_loss_curves(), config=dict(displayModeBar=False), style=dict(height=300))),
                card(dcc.Graph(figure=fig_clusters(),    config=dict(displayModeBar=False), style=dict(height=300))),
            ]),
            model_selector,
        ])

    # FEATURES
    elif tab == "features":
        code_snippet = """# Feature Importance with scikit-learn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# 1. Prepare data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)

# 2. Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_s, y_train)

# 3. Extract and sort importances
importances = pd.Series(
    rf.feature_importances_, index=feature_names
).sort_values(ascending=False)

print(importances)

# 4. Visualise with Plotly
fig = px.bar(importances, orientation='h',
             title='Feature Importance (Gini)',
             labels={'value': 'Importance', 'index': 'Feature'})
fig.show()
"""
        return html.Div([
            html.Div(style=dict(display="grid", gridTemplateColumns="1fr 1fr", gap=16), children=[
                card(dcc.Graph(figure=fig_feature_importance(),
                               config=dict(displayModeBar=False), style=dict(height=340))),
                card([
                    section_label("Python Code — Feature Importance"),
                    html.Pre(code_snippet, style=dict(
                        background="rgba(2,8,23,0.85)", border=f"1px solid {BORDER}",
                        borderRadius=10, padding=16, fontSize=11, lineHeight=1.8,
                        color="#e2e8f0", overflowX="auto", margin=0,
                        maxHeight=320, overflowY="auto",
                    )),
                ]),
            ]),
        ])

    # APPLY 
    elif tab == "apply":
        role_cards = html.Div(style=dict(display="grid", gridTemplateColumns="repeat(3,1fr)", gap=14, marginBottom=20), children=[
            card([
                html.P("OPEN ROLE", style=dict(color=ACCENT, fontSize=9, letterSpacing=3, margin="0 0 6px")),
                html.H3(role, style=dict(color=TEXT, fontSize=15, margin="0 0 8px", fontWeight=600)),
                html.P("Remote · Full-time · Competitive package",
                       style=dict(color=MUTED, fontSize=11, margin=0)),
            ])
            for role in ROLES[:3]
        ])

        form = card([
            section_label("Apply for a Role"),
            html.Div(style=dict(display="grid", gridTemplateColumns="1fr 1fr", gap=14), children=[
                html.Div([html.Label("Full Name *", style=dict(color=MUTED, fontSize=11, display="block", marginBottom=6)),
                          input_field("name", "Jane Smith")]),
                html.Div([html.Label("Email *", style=dict(color=MUTED, fontSize=11, display="block", marginBottom=6)),
                          input_field("email", "jane@email.com", "email")]),
                html.Div([html.Label("Role *", style=dict(color=MUTED, fontSize=11, display="block", marginBottom=6)),
                          dcc.Dropdown(id="apply-role",
                              options=[{"label": r, "value": r} for r in ROLES],
                              placeholder="Select a role…",
                              style=dict(background="rgba(15,23,42,0.9)", fontFamily="monospace", fontSize=12))]),
                html.Div([html.Label("Years of Experience *", style=dict(color=MUTED, fontSize=11, display="block", marginBottom=6)),
                          dcc.Dropdown(id="apply-exp",
                              options=[{"label": v, "value": v} for v in ["0–1 yrs","1–3 yrs","3–5 yrs","5+ yrs"]],
                              placeholder="Select…",
                              style=dict(background="rgba(15,23,42,0.9)", fontFamily="monospace", fontSize=12))]),
                html.Div(style=dict(gridColumn="1 / -1"), children=[
                    html.Label("GitHub / Portfolio", style=dict(color=MUTED, fontSize=11, display="block", marginBottom=6)),
                    input_field("github", "https://github.com/yourprofile"),
                ]),
                html.Div(style=dict(gridColumn="1 / -1"), children=[
                    html.Label("Cover Note", style=dict(color=MUTED, fontSize=11, display="block", marginBottom=6)),
                    dcc.Textarea(id="cover-note", placeholder="Tell us about your ML/AI background…",
                        style=dict(width="100%", minHeight=90, background="rgba(15,23,42,0.9)",
                                   border=f"1px solid {BORDER}", borderRadius=8, color=TEXT,
                                   padding="10px 14px", fontSize=12, fontFamily="monospace",
                                   resize="vertical", boxSizing="border-box")),
                ]),
            ]),
            html.Div(style=dict(marginTop=16, display="flex", gap=12), children=[
                html.Button("Submit Application →", id="submit-btn", n_clicks=0,
                    style=dict(flex=1, background=f"linear-gradient(135deg, #0ea5e9, {PURPLE})",
                               border="none", borderRadius=8, color="#fff",
                               padding="12px 0", cursor="pointer", fontFamily="monospace",
                               fontSize=13, fontWeight=700, letterSpacing=1)),
                html.Button("Clear", id="clear-btn", n_clicks=0,
                    style=dict(background="transparent", border=f"1px solid {BORDER}",
                               borderRadius=8, color=MUTED, padding="12px 20px",
                               cursor="pointer", fontFamily="monospace", fontSize=12)),
            ]),
            html.Div(id="form-feedback", style=dict(marginTop=12)),
        ])

        return html.Div([role_cards, form])

    return html.Div("Tab not found.")



#  CALLBACKS
@app.callback(
    Output("conf-matrix", "children"),
    Input("model-dd", "value"),
)
def update_confusion(model_name):
    if not model_name:
        return html.Div()
    return dcc.Graph(figure=fig_confusion(model_name),
                     config=dict(displayModeBar=False), style=dict(height=280))


@app.callback(
    Output("form-feedback", "children"),
    Input("submit-btn", "n_clicks"),
    State("name", "value"),
    State("email", "value"),
    State("apply-role", "value"),
    State("apply-exp", "value"),
    State("github", "value"),
    State("cover-note", "value"),
    prevent_initial_call=True,
)
def submit_application(n, name, email, role, exp, github, note):
    errors = []
    if not name or not name.strip():
        errors.append("• Full Name is required.")
    if not email or "@" not in email:
        errors.append("• A valid email is required.")
    if not role:
        errors.append("• Please select a role.")
    if not exp:
        errors.append("• Please select years of experience.")

    if errors:
        return html.Div([html.P(e, style=dict(color="#f43f5e", fontSize=12, margin="2px 0")) for e in errors])

    return html.Div(style=dict(
        background="rgba(56,189,248,0.08)", border=f"1px solid {BORDER}",
        borderRadius=10, padding="16px 20px",
    ), children=[
        html.Span("  ", style=dict(fontSize=18)),
        html.Span(f"Thanks, {name}! Your application for ", style=dict(color=TEXT, fontSize=13)),
        html.Strong(role, style=dict(color=ACCENT)),
        html.Span(f" has been received. We'll be in touch at ", style=dict(color=TEXT, fontSize=13)),
        html.Strong(email, style=dict(color=PINK)),
        html.Span(".", style=dict(color=TEXT)),
    ])

# ------------------------------------------------------------#
#  RUN
if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════════╗")
    print("║  ML / AI Dashboard  →  http://127.0.0.1:8050   ║")
    print("╚══════════════════════════════════════════════════╝\n")
    app.run(debug=True, port=8050)
