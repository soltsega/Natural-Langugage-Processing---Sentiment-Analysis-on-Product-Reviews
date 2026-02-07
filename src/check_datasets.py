import pandas as pd
import re
import os

def check_amazon():
    path = os.path.join('data', 'amazon_reviews.csv')
    if not os.path.exists(path):
        return "Amazon file not found."
    
    df = pd.read_csv(path, low_memory=False)
    text_col = 'reviews.text'
    rating_col = 'reviews.rating'
    
    stats = {
        'Dataset': 'Amazon Product Reviews',
        'Total Rows': len(df),
        'Text Missing': df[text_col].isna().sum(),
        'Rating Missing': df[rating_col].isna().sum(),
        'Unique Ratings': sorted(df[rating_col].dropna().unique()),
        'Rating Distribution': df[rating_col].value_counts(normalize=True).to_dict(),
        'Avg Word Count': df[text_col].dropna().apply(lambda x: len(str(x).split())).mean(),
        'Has HTML': df[text_col].dropna().apply(lambda x: bool(re.search('<.*?>', str(x)))).sum()
    }
    return stats

def check_imdb():
    path = os.path.join('data', 'imdb_reviews.csv')
    if not os.path.exists(path):
        return "IMDb file not found."
    
    df = pd.read_csv(path)
    text_col = 'review'
    label_col = 'sentiment'
    
    stats = {
        'Dataset': 'IMDb Movie Reviews',
        'Total Rows': len(df),
        'Text Missing': df[text_col].isna().sum(),
        'Rating Missing': df[label_col].isna().sum(),
        'Unique Ratings': sorted(df[label_col].dropna().unique()),
        'Rating Distribution': df[label_col].value_counts(normalize=True).to_dict(),
        'Avg Word Count': df[text_col].dropna().apply(lambda x: len(str(x).split())).mean(),
        'Has HTML': df[text_col].dropna().apply(lambda x: bool(re.search('<.*?>', str(x)))).sum()
    }
    return stats

if __name__ == "__main__":
    print("-" * 30)
    print("DATASET COMPARISON")
    print("-" * 30)
    
    amazon = check_amazon()
    imdb = check_imdb()
    
    for stat in [amazon, imdb]:
        if isinstance(stat, dict):
            print(f"\n{stat['Dataset']}:")
            for k, v in stat.items():
                if k != 'Dataset':
                    print(f"  {k}: {v}")
        else:
            print(stat)
