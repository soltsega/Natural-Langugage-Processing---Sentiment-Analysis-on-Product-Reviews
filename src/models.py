"""
Model Architectures and Pipeline Definitions for Sentiment Analysis.

This module provides standardized sklearn Pipeline factory functions
used throughout the project for model training and evaluation.
"""

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

def get_text_transformer():
    """
    Returns a text feature extraction pipeline (TF-IDF + SVD + Scaling).
    """
    return Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english', min_df=5)),
        ('svd', TruncatedSVD(n_components=100, random_state=42)),
        ('scaler', StandardScaler())
    ])

def get_logreg_smote_pipeline():
    """
    Returns the production-ready Logistic Regression + SMOTE pipeline.
    
    Returns:
        ImbPipeline: The complete processing and classification pipeline.
    """
    # Note: ColumnTransformer setup depends on dataframe schema (brand, categories, cleaned_text)
    # This is a factory function for the core architecture.
    
    text_pipe = get_text_transformer()
    
    preprocessor = ColumnTransformer([
        ('text', text_pipe, 'cleaned_text'),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['brand', 'categories'])
    ])
    
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('clf', LogisticRegression(max_iter=2000, n_jobs=-1, random_state=42))
    ])
    
    return pipeline
