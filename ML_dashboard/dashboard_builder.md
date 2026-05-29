## ML / AI & Data Science Dashboard 

A comprehensive walkthrough of the Python + Dash dashboard

---

### Table of Contents

1. Dashboard  Descriptions
2. Tech Stack
3. How to Run the Dashboard
4. Dashboard Structure (Tabs)
5. ML Pipeline — Step by Step
6. ML Model Descriptions
7. Visualization Details
8. Interactive Callbacks
9. Key Concepts Glossary

---

#### 1. Dashboard  Descriptions

This is an **interactive Machine Learning experiment tracker** built with Python and Dash by Plotly. It simulates a real-world ML workflow:

- Generate data
- Train multiple classification models
- Compare their performance
- Visualize results with charts
- Optionally submit a job application form

Everything runs locally in browser at `http://127.0.0.1:8050`.

---

#### 2. Tech Stack

| Library | Role |
|---|---|
| `dash` | Web framework — builds the interactive UI in Python |
| `plotly` | Charts and graphs rendered in the browser |
| `numpy` | Numerical computations and random data generation |
| `pandas` | Tabular data/DataFrames for metrics |
| `scikit-learn` | All ML models, preprocessing, and evaluation metrics |

---

#### 3. How to Run the Dashboard

```bash
# Step 1 — Install dependencies
pip install dash plotly pandas numpy scikit-learn

# Step 2 — Run the app
python ml_dashboard.py

# Step 3 — Open in browser
# Go to: http://127.0.0.1:8050
```

---

#### 4. Dashboard Structure (Tabs)

The UI is divided into 4 tabs, each serving a different purpose:

#### Tab 1 — Overview
- Shows KPI metric cards (best accuracy, sample count, features, models)
- Bar chart comparing model accuracy and AUC-ROC
- Skill radar chart
- ROC curves for all 3 models
- Summary table with all metrics

#### Tab 2 — Training
- Training & Validation loss curves (simulated over 20 epochs)
- K-Means cluster scatter plot
- Dropdown to select a model -> displays its Confusion Matrix

#### Tab 3 — Features
- Horizontal bar chart of feature importances from the Random Forest
- Code snippet showing how feature importance is computed

#### Tab 4 — Apply
- A mock job application form for ML/AI roles
- Form validation (name, email, role, experience required)
- Styled success/error feedback messages

---

#### 5. ML Pipeline — Step by Step

This is the core ML workflow in the code. Each step mirrors what you'd do in a real project.

---

#### Step 1 — Generate Synthetic Data

```python
X, y = make_classification(
    n_samples=600, n_features=10,
    n_informative=6, n_redundant=2,
    random_state=42
)
```

The above  creates a fake binary classification dataset with 600 samples and 10 features. 6 features carry real signal (`n_informative`), 2 are linear combinations of others (`n_redundant`), and 2 are noise.

 Here the real-world data can't be shipped with a demo. Synthetic data lets the dashboard be fully self-contained while still demonstrating realistic ML behavior.

The features are given human-readable names to make the charts meaningful:
```
Age, Income, Tenure, Usage, Region, Device, Plan, Visits, Spend, Churn_Hist
```

---

#### Step 2 — Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
```

This splits 600 samples into 450 for training (75%) and 150 for testing (25%). Because we must never evaluate a model on the same data it was trained on — that would give falsely high scores. The test set simulates "unseen" real-world data. `random_state=42` ensures the same split every run.

---

#### Step 3 — Feature Scaling (StandardScaler)

```python
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
```

This transforms each feature to have mean = 0 and standard deviation = 1. Logistic Regression and Gradient Boosting are sensitive to feature scale. Without scaling, a feature with large values (e.g., Income = 50,000) would dominate features with small values (e.g., Visits = 3). It is important to  fit the scaler **only** on training data, then apply it to test data — to avoid data leakage.

---

#### Step 4 — Train Three Models

Three classifiers are trained in a loop:

```python
models = {
    "Logistic Regression": LogisticRegression(...),
    "Random Forest":       RandomForestClassifier(...),
    "Gradient Boosting":   GradientBoostingClassifier(...),
}
for name, mdl in models.items():
    mdl.fit(X_train_s, y_train)
```

Each model learns to map the 10 features → binary label (0 or 1). See section 6 for why each model is included.

---

#### Step 5 — Evaluate: Accuracy & AUC-ROC

```python
acc     = mdl.score(X_test_s, y_test)
y_prob  = mdl.predict_proba(X_test_s)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
```

**Accuracy** = proportion of correct predictions. Simple but can be misleading on imbalanced data.

**AUC-ROC** (Area Under the ROC Curve) = measures how well the model ranks positives above negatives across all thresholds. A score of 1.0 is perfect; 0.5 is random guessing. More robust than raw accuracy.

---

#### Step 6 — Feature Importance

```python
importances = pd.Series(
    rf.feature_importances_, index=FEATURE_NAMES
).sort_values()
```

 The Random Forest internally tracks how much each feature reduced impurity (Gini index) across all its decision trees.

 This tells us which input variables matter most for prediction — useful for model explainability and feature selection in real projects.

---

#### Step 7 — Confusion Matrix

```python
y_pred = mdl.predict(X_test_s)
cm = confusion_matrix(y_test, y_pred)
```

A 2×2 matrix showing:

|  | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | True Negative (TN) | False Positive (FP) |
| **Actual 1** | False Negative (FN) | True Positive (TP) |

 Accuracy alone hides the types of errors. A confusion matrix reveals whether your model is more likely to produce false alarms (FP) or miss real positives (FN) — critical in domains like fraud detection or medical diagnosis.

---

#### Step 8 — Simulated Training Curves

```python
train_loss = [0.95 * np.exp(-0.18 * e) + noise for e in epochs]
val_loss   = [0.98 * np.exp(-0.15 * e) + noise + 0.03 for e in epochs]
```

The above simulates how loss would decrease over 20 training epochs using an exponential decay formula plus random noise. The sklearn models used here don't produce epoch-by-epoch curves. The simulation demonstrates what these curves look like in practice — the validation loss being slightly higher than training loss is realistic and expected.

---

#### Step 9 — Cluster Visualization

```python
X_blob, y_blob = make_blobs(
    n_samples=300, centers=3,
    cluster_std=1.4, random_state=7
)
```

The above generates 3 clusters of 2D data points, simulating the output of a K-Means clustering algorithm.  Demonstrates unsupervised learning (no labels needed) — useful for customer segmentation, anomaly detection, etc.

---

#### 6. ML Model Descriptions

#### Logistic Regression
- Linear classifier
- Fits a line (in high dimensions, a hyperplane) that separates the two classes. Outputs a probability using the sigmoid function.
- Fast, interpretable, great baseline
- Establishes a baseline. If more complex models don't beat it by much, simpler is better.

---

#### Random Forest
- Ensemble of Decision Trees
- Builds 100 decision trees on random subsets of data and features, then averages their predictions (bagging).
- Handles non-linear patterns, resistant to overfitting, provides feature importance
- Often the best out-of-the-box performer. Also used for the feature importance chart.

---

#### Gradient Boosting
- Ensemble (boosting)
- Builds trees sequentially — each new tree corrects the errors of the previous one by fitting residuals.
- Usually highest accuracy among classical ML models
- Represents state-of-the-art classical ML performance 

---

#### 7. Visualization Details

| Chart |  Visualization | Usefulness |
|---|---|---|
| **Model Comparison Bar Chart** | Accuracy % and AUC-ROC % side by side | Quick performance comparison across models |
| **ROC Curves** | True Positive Rate vs False Positive Rate at every threshold | Shows overall discrimination ability; AUC summarizes it in one number |
| **Confusion Matrix** | TP / TN / FP / FN counts | Reveals the type of errors a model makes |
| **Feature Importance** | Which features drive predictions most | Model explainability; guides feature engineering |
| **Training/Validation Loss** | How loss changes each epoch | Detects overfitting (val loss rising while train loss falls) |
| **Cluster Scatter Plot** | 2D view of 3 data clusters | Visualizes unsupervised groupings |
| **Skill Radar Chart** | Personal skill scores across 7 domains | Visual CV / profile summary |

---

#### 8. Interactive Callbacks

Dash uses `@app.callback` decorators to connect UI components to Python functions without page reloads.

| Trigger | Output | Outcome |
|---|---|---|
| Tab click | Tab content div | Entire tab layout is re-rendered |
| Model dropdown change | Confusion matrix div | Recomputes and displays confusion matrix for selected model |
| Submit button click | Feedback div | Validates form fields; shows error messages or success banner |

---

#### 9. Key Concepts Glossary

| Term | Descriptions |
|---|---|
| **Classification** | Predicting which category something belongs to (e.g., yes/no) |
| **Train/Test Split** | Holding back data to evaluate how well the model generalises |
| **StandardScaler** | Normalising features so no single feature dominates due to its scale |
| **AUC-ROC** | A score (0–1) measuring how well the model separates the two classes |
| **Feature Importance** | How much each input variable contributes to the model's decisions |
| **Overfitting** | Model memorises training data and fails on new data (val loss > train loss) |
| **Ensemble** | Combining many weak models (trees) into one strong model |
| **Bagging** | Training models on random subsets independently, then averaging (Random Forest) |
| **Boosting** | Training models sequentially, each fixing the last one's mistakes (Gradient Boosting) |
| **Confusion Matrix** | Table showing counts of correct and incorrect predictions by class |

---


