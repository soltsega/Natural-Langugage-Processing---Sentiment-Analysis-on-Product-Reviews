# Dataset Selection & Comparative Analysis

This document details the comparative analysis of the Amazon Product Reviews and IMDb Movie Reviews datasets and explains the rationale for selecting the Amazon dataset for this project.

## 1. Comparative Analysis Breakdown

We evaluated both datasets against standard NLP and Data Science requirements to determine which would provide the best learning and implementation experience.

| Feature | Amazon Product Reviews | IMDb Movie Reviews |
| :--- | :--- | :--- |
| **Industry Context** | E-commerce / Retail Analytics | Entertainment / Sentiment Benchmarking |
| **Total Record Count** | 34,660 | **50,000** |
| **Metadata Richness** | **High** (21 Fields: Brand, Category, Date, Verified) | **Low** (2 Fields: Text, Sentiment) |
| **Text Complexity** | Short, punchy, real-world vernacular | Long, descriptive, consistent narrative |
| **Sentiment Type** | 5-Class Rating (1-5 Stars) | Binary (Positive/Negative) |
| **Class Balance** | **Highly Imbalanced** (68% are 5-star) | **Perfectly Balanced** (50/50 split) |
| **Data Cleanliness** | Clean text, no HTML noise | Significant HTML noise (`<br />` tags) |

## 2. Rational for Choosing Amazon Product Reviews

While the IMDb dataset is a popular "clean" benchmark for sentiment analysis, we have selected the **Amazon Product Reviews** dataset primarily for its "Real-World" complexity.

### A. Handling Class Imbalance
The Amazon dataset is skewed towards positive reviews. In a professional setting, data is rarely balanced.
- **Learning Outcome**: We will implement advanced techniques like **Class Weighting**, **Synthetic Minority Over-sampling (SMOTE)**, or **Precision-Recall thresholding** to ensure the model accurately detects negative sentiment despite the imbalance.

### B. Exploiting Structured Metadata
The availability of 21 metadata fields allows us to perform significantly more advanced EDA and feature engineering.
- **Learning Outcome**: We can analyze how sentiment varies across different **Product Categories** or **Brands**. We can also determine if **Verified Purchases** correlate with higher or lower sentiment scores, providing a dual perspective of NLP and Business Intelligence.

### C. Scalability and Flexibility
The 5-star rating system provides the option to move from simple binary classification to **Multi-Class Classification** or even **Regression** (predicting the exact star rating), which is a common requirement in industry-grade product analytics.

## 3. Final Decision
**Selected Dataset:** `data/amazon_reviews.csv`

The Amazon dataset provides a richer, more challenging, and ultimately more educational environment that better mirrors the complexity of industrial Data Science projects.

---
