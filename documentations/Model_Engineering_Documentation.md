# Documentation: Model Engineering Phase

## 1. Overview
The **Model Engineering Phase** (documented in `02_model_engineering.ipynb`) focuses on building, optimizing, and evaluating machine learning pipelines for sentiment analysis of product reviews. The core goal is to classify reviews into three categories:
- **0**: Negative (1-2 Stars)
- **1**: Neutral (3 Stars)
- **2**: Positive (4-5 Stars)

## 2. Data Pipeline
- **Dataset**: `cleaned_amazon.csv` (preprocessed in Phase 1/2).
- **Target Engineering**: Star ratings are binned into the three classes above.
- **Split Strategy**: Stratified 80/20 train-test split to preserve class distribution.

## 3. Feature Engineering Architecture
The project utilizes a `ColumnTransformer` to handle multi-modal features:
- **Text Features (`cleaned_text`)**:
    - `TfidfVectorizer`: (Max 5000 features, n-grams 1-2).
    - `TruncatedSVD`: Dimensionality reduction to 100 components.
    - `StandardScaler`: Normalizing latent features.
- **Categorical Features (`brand`, `categories`)**:
    - `OneHotEncoder`: Handling unknown categories via `handle_unknown='ignore'`.

## 4. Modeling Suite

### 4.1. Baseline: Multinomial Naive Bayes
- Simple, high-speed baseline.
- Uses a simplified preprocessor (CountVectorizer/TF-IDF) without SVD.

### 4.2. Weighted Logistic Regression
- **Optimization**: Uses `class_weight='balanced'` to automatically penalize majority class errors.
- **Performance**: High reliability for a linear model.

### 4.3. XGBoost
- **Configuration**: Uses `tree_method='hist'` for speed.
- **Focus**: Non-linear patterns in the text and metadata.

### 4.4. SMOTE Pipeline (`imblearn`)
- **Strategy**: Synthetic Minority Over-sampling Technique (SMOTE) is integrated into the `ImbPipeline` to ensure oversampling occurs only within training folds, preventing data leakage.

## 5. Evaluation & Diagnostics

### 5.1. Performance Comparison
Models are compared using **F1-Score (Macro)** to ensure performance on minority classes is weighed equally.

| Model | F1 (Macro) | Note |
| :--- | :--- | :--- |
| **LogReg (SMOTE)** | ~0.40 | Best overall balance. |
| **LogReg (Weighted)**| ~0.39 | Robust baseline. |
| **XGBoost** | ~0.36 | Room for hyperparameter tuning. |

### 5.2. Robustness Check
- **5-Fold Cross-Validation**: Performed on the Logistic Regression pipeline to verify stability across data folds.

### 5.3. Diagnostic Tools
- **Misclassification Audit**: Manual inspection of "Severe Errors" (Actual Negative predicted as Positive).
- **Bias Analysis**: Accuracy breakdown by Brand to audit for model fairness.

## 6. Visual Validation (Phase 4)
A `plot_confusion_matrices` utility is provided to generate a 2x2 grid of Heatmaps.
- **Normalization**: Normalized by row (Recall) to highlight class-specific sensitivity.
- **Goal**: Identify confusion clusters between Neutral and Negative classes.

---
