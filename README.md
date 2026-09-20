# 🔬 SignalScope AI

**Explainable machine learning for Higgs signal classification using physics-engineered features.**

🌐 **Live Demo:** https://signalscope-ai.streamlit.app/

---

## Overview

SignalScope AI is an AI + particle physics project that explores how
physics-engineered features affect machine-learning-based Higgs signal
classification.

The project uses the HIGGS dataset to distinguish simulated Higgs signal
events from background events and investigates whether a compact set of
physically derived variables can improve classification performance.

An interactive Streamlit application allows users to enter seven high-level
physics features and receive a Higgs signal probability together with
model insights.

---

## Problem

Particle collision experiments generate large amounts of complex data.

Machine learning can help distinguish potentially interesting signal events
from background processes. However, an important question is whether raw
detector-level information is sufficient, or whether features engineered
using physics knowledge provide additional discriminative information.

SignalScope AI explores the question:

> How much do physics-engineered high-level features contribute to
> machine-learning-based Higgs signal classification?

---

## Dataset

This project uses a 10,000-event subset of the **HIGGS dataset** from the
UCI Machine Learning Repository.

The original dataset contains:

- 21 low-level kinematic features
- 7 high-level features engineered by physicists
- Binary labels:
  - `1` → signal
  - `0` → background

The seven high-level variables used in the interactive demo are:

`m_jj`, `m_jjj`, `m_lv`, `m_jlv`, `m_bb`, `m_wbb`, `m_wwbb`

Dataset:
https://archive.ics.uci.edu/dataset/280/higgs

---

## Machine Learning Pipeline

Two main classifiers were explored:

- Logistic Regression as a baseline
- Random Forest for improved nonlinear classification

Random Forest was then used to study three different feature groups:

1. Low-Level Only — 21 features
2. High-Level Only — 7 features
3. All Features — 28 features

The same train/test split and model configuration were used for the
feature-group comparison.

---

## Results

| Feature Set | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Low-Level Only (21) | 0.621 | 0.626 | 0.705 | 0.663 | 0.665 |
| High-Level Only (7) | 0.685 | 0.690 | 0.735 | 0.712 | 0.758 |
| All Features (28) | **0.716** | **0.717** | **0.767** | **0.741** | **0.793** |

The seven high-level physics features alone outperform the 21 low-level
features despite using a much smaller representation.

The strongest performance is obtained when both feature groups are
combined, suggesting that low-level and high-level variables contain
complementary information.

---

## Explainability

Random Forest feature importance was used to investigate which variables
contribute most strongly to classification.

Among the high-level features, `m_bb` emerged as the most influential
variable in the trained model.

This provides an interpretable connection between the machine-learning
model and physics-engineered event representations.

---

## Interactive Demo

The Streamlit application uses the **High-Level Only Random Forest model**
with seven input features.

Users can:

- enter collision-event feature values
- calculate Higgs signal probability
- view the predicted signal/background class
- compare ROC-AUC across feature groups
- inspect high-level feature importance

### Try SignalScope AI

https://signalscope-ai.streamlit.app/

> The deployed demo uses the 7-feature model with ROC-AUC 0.758.
> The 28-feature model achieved ROC-AUC 0.793 in the experimental notebook.

---

## Tech Stack

- Python
- pandas
- scikit-learn
- matplotlib
- joblib
- Streamlit
- Google Colab
- Git / GitHub

---

## Repository Structure

```text
signalscope-ai/
│
├── 01_baseline_model.ipynb
├── app.py
├── rf_high_model.pkl
├── requirements.txt
├── README.md
└── data/
```

## Run Locally

Clone the repository:

```bash
git clone https://github.com/numericalmind/signalscope-ai.git
cd signalscope-ai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## Limitations

SignalScope AI is an educational and experimental prototype.

The current experiments use a 10,000-event subset of the full HIGGS dataset, and the deployed demo uses only the seven high-level features.

The predictions should therefore not be interpreted as results from a real particle-physics analysis pipeline.

---

## Future Work

Potential extensions include:

- training on a larger portion of the HIGGS dataset
- testing additional machine-learning models
- adding SHAP-based explainability
- investigating classification thresholds
- extending the interactive physics visualizations
- comparing learned representations with physics-engineered features

---

## Project Goal

SignalScope AI demonstrates how domain knowledge and machine learning can work together in scientific classification problems.

Rather than treating particle-physics data as a generic ML dataset, the project focuses on understanding the contribution of physically motivated features to model performance and interpretability.