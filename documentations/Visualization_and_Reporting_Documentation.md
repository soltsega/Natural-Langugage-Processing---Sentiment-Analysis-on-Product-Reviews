# Documentation: Visualization & Reporting (Phase 5)

## 1. Objective
Phase 5 focuses on transforming raw model outputs into actionable visual insights. The goal is to create intuitive dashboards that stakeholders can use to understand model performance and sentiment drivers without needing deep technical knowledge.

## 2. Visualizations Implemented

### 2.1 Sentiment WordClouds
We generated high-resolution WordClouds to visualize the most significant terms for each sentiment class.
- **Positive Reviews (4-5 Stars)**: Dominated by terms like "great," "love," "easy," and "perfect," indicating satisfaction with product utility and usability.
- **Negative Reviews (1-2 Stars)**: Characterized by words like "disappointed," "broke," "waste," and "money," highlighting issues with quality control and value.

### 2.2 Interactive Model Dashboard
We moved beyond static charts to interactive dashboards using **Seaborn** and **Matplotlib** (used as a robust fallback for Plotly).
- **Performance Metrics**: A grouped bar chart displaying Precision, Recall, and F1-Score for each class side-by-side. This clearly visualizes the performance gap between the majority class (Positive) and minority classes.
- **Confusion Matrix**: A heatmap providing a granular view of misclassifications. It highlights that the model most frequently confuses "Neutral" reviews with "Positive" ones, a common challenge in Sentiment Analysis.

## 3. Code Audit & Modularization
To ensure the codebase is maintainable and production-ready, we transitioned from a notebook-centric workflow to a modular Python package structure in `src/`.

| Module | Functionality |
| :--- | :--- |
| **`check_datasets.py`** | Diagnostic tools to inspect rows, missing values, and rating distributions of raw data. |
| **`preprocessing.py`** | Contains the `CleanText` transformer for cleaning raw text and `bin_sentiment` for target encoding. |
| **`models.py`** | Defines standard sklearn Pipelines, including the `LogisticRegression` + `SMOTE` architecture. |

## 4. Final Project Status
With the completion of visualization and documentation, the sentiment analysis pipeline is now fully operational.
- **Input**: Raw Amazon product reviews.
- **Processing**: Automated cleaning and TF-IDF vectorization.
- **Model**: Logistic Regression with SMOTE to handle imbalance.
- **Output**: Sentiment prediction (Negative/Neutral/Positive) with associated probability scores.

---
**Status**: [COMPLETED]
