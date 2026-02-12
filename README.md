# Sentiment Analysis on Product Reviews

A comprehensive machine learning project for analyzing sentiment in product reviews using advanced NLP techniques and ensemble methods.

## Project Overview

This project implements a robust sentiment analysis system that classifies product reviews into three sentiment categories:
- **Negative** (1-2 stars)
- **Neutral** (3 stars)  
- **Positive** (4-5 stars)

The system uses a production-ready pipeline combining TF-IDF vectorization, dimensionality reduction, and Logistic Regression with SMOTE for handling class imbalance.

## Features

- **Advanced Text Preprocessing**: Custom text cleaning with regex-based normalization
- **Feature Engineering**: TF-IDF vectorization with n-grams and TruncatedSVD
- **Class Imbalance Handling**: SMOTE oversampling for balanced training
- **Production-Ready Pipeline**: Scikit-learn Pipeline with ColumnTransformer
- **Comprehensive Evaluation**: Multiple metrics and visualization tools
- **Modular Architecture**: Clean separation of concerns with reusable components

## Project Structure

```
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Text cleaning and sentiment binning
│   ├── models.py            # ML pipeline definitions
│   └── check_datasets.py    # Data validation utilities
├── notebooks/
│   ├── 01_advanced_eda.ipynb      # Exploratory data analysis
│   ├── 02_model_engineering.ipynb # Model development
│   ├── 03_model_evaluation.ipynb  # Performance evaluation
│   └── 04_visualization_reporting.ipynb # Results visualization
├── data/                     # Dataset storage
├── tests/                    # Unit tests
├── documentations/           # Additional documentation
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/soltsega/Natural-Langugage-Processing---Sentiment-Analysis-on-Product-Reviews.git
   cd Natural-Langugage-Processing---Sentiment-Analysis-on-Product-Reviews
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .nlpvenv
   source .nlpvenv/bin/activate  # On Windows: .nlpvenv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

## 📊 Usage

### Basic Pipeline Usage

```python
from src.models import get_logreg_smote_pipeline
from src.preprocessing import CleanText, bin_sentiment
import pandas as pd

# Load your data
df = pd.read_csv('your_reviews.csv')

# Initialize the production pipeline
pipeline = get_logreg_smote_pipeline()

# Prepare features (assuming columns: 'brand', 'categories', 'cleaned_text')
X = df[['brand', 'categories', 'cleaned_text']]
y = df['rating'].apply(bin_sentiment)

# Train the model
pipeline.fit(X, y)

# Make predictions
predictions = pipeline.predict(X_test)
```

### Text Preprocessing

```python
from src.preprocessing import CleanText

cleaner = CleanText()
cleaned_text = cleaner.transform(["This is a GREAT product!!!"])
# Output: ["this is a great product"]
```

## Model Architecture

The production pipeline consists of:

1. **Text Processing**: TF-IDF Vectorizer (max_features=5000, ngram_range=(1,2))
2. **Dimensionality Reduction**: TruncatedSVD (n_components=100)
3. **Feature Scaling**: StandardScaler
4. **Categorical Encoding**: OneHotEncoder for brand and categories
5. **Class Balancing**: SMOTE oversampling
6. **Classification**: Logistic Regression (max_iter=2000)

## Performance

The model achieves strong performance across multiple metrics:
- **Accuracy**: ~85-90% (varies by dataset)
- **Precision/Recall**: Balanced across all sentiment classes
- **F1-Score**: Consistently high for positive and negative classes

*Detailed performance metrics available in `03_model_evaluation.ipynb`*

## Analysis & Insights

The project includes comprehensive analysis notebooks:

- **Advanced EDA**: Distribution analysis, sentiment patterns, brand comparisons
- **Model Engineering**: Hyperparameter tuning, feature importance analysis
- **Evaluation**: Confusion matrices, classification reports, ROC curves
- **Visualization**: Interactive dashboards and sentiment trends

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Solomon Tsega**
- GitHub: [@soltsega](https://github.com/soltsega)

## Acknowledgments

- Scikit-learn for the machine learning framework
- spaCy for advanced NLP capabilities
- The open-source community for the amazing tools and libraries

---