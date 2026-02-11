"""
Preprocessing Utilities for Sentiment Analysis Project.

This module contains classes and functions for cleaning raw review text
and engineering features for machine learning pipelines.
"""

import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CleanText(BaseEstimator, TransformerMixin):
    """
    Custom Transformer for cleaning text data.
    Perform lowercase, special character removal, and whitespace normalization.
    """
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Cleans the input text data.
        
        Args:
            X (pd.Series or list): Uncleaned text strings.
            
        Returns:
            pd.Series: Cleaned text strings.
        """
        if isinstance(X, list):
            X = pd.Series(X)
            
        X = X.astype(str).str.lower()
        X = X.apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
        X = X.apply(lambda x: re.sub(r'\s+', ' ', x).strip())
        return X

def bin_sentiment(rating):
    """
    Convert 1-5 star ratings into 3 sentiment classes.
    
    Args:
        rating (int or float): Star rating.
        
    Returns:
        int: 0 (Negative), 1 (Neutral), 2 (Positive).
    """
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2
