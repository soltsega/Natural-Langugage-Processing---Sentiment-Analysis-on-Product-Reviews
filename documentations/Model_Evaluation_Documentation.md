# Documentation: Model Evaluation Phase (Phase 4)

## 1. Overview
The **Model Evaluation Phase** (documented in `03_model_evaluation.ipynb`) moves beyond simple accuracy to rigorously validate the performance of our sentiment analysis pipeline. We focus on handling class imbalance and understanding specific failure modes.

## 2. Advanced Metrics
To ensure the model performs well across all classes (Negative, Neutral, Positive), we use metrics that are sensitive to class distribution:

- **Balanced Accuracy**: Arithmetic mean of class-specific recall. Ensures minority classes are given equal weight.
- **F1-Score (Macro)**: The harmonic mean of precision and recall, averaged across classes.
- **ROC-AUC (One-vs-Rest)**: Measures the model's ability to distinguish between any one class and the others. This is independent of the classification threshold.

| Metric | Score (Target) | Note |
| :--- | :--- | :--- |
| **ROC-AUC (OvR)** | > 0.85 | High discriminative power. |
| **F1 (Macro)** | ~0.40 | Reflects difficulty of the Neutral class. |
| **Balanced Accuracy**| ~0.45 | Significantly better than random (0.33). |

## 3. Threshold Tuning
For sentiment analysis, correctly identifying **Negative** reviews (Class 0) is often more critical than "Neutral" or "Positive" labels.

- **Experiment**: We lowered the prediction threshold for the Negative class to **0.4**.
- **Result**: Increased Recall for Negatives at the cost of some Precision. This allows us to catch more dissatisfied customers even if the model is only 40% confident.

## 4. Error Analysis Findings
Through manual inspection of high-confidence misclassifications, we identified key patterns:
1. **Sarcasm**: The model often fails on sarcastic reviews (e.g., "Great, another broken part").
2. **Neutral Ambiguity**: Many 3-star reviews contain both strong "positive" and "negative" keywords, causing confusion.
3. **Short Text**: Very short reviews lack sufficient context for TF-IDF to generate strong features.

## 5. Next Steps (Phase 5)
- **Advanced Visualization**: Plotting ROC and Precision-Recall curves.
- **Reporting**: Compiling an executive summary of model readiness.
- **Final Export**: Saving the trained pipeline and metadata for production-ready status.

---
