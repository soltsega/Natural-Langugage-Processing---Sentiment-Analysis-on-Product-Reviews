import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load Data
DATA_PATH = "data/amazon_reviews.csv"
VIS_PATH = "visualizations/"

if not os.path.exists(VIS_PATH):
    os.makedirs(VIS_PATH)

print("Loading dataset...")
if not os.path.exists(DATA_PATH):
    print(f"Error: {DATA_PATH} not found!")
else:
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Select relevant columns
    df = df[['reviews.text', 'reviews.rating']]
    df.dropna(inplace=True)

    print(f"Dataset Shape: {df.shape}")

    # 1. Class Distribution (Ratings)
    plt.figure(figsize=(10, 6))
    sns.countplot(x='reviews.rating', data=df, palette='viridis')
    plt.title('Distribution of Product Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.savefig(f"{VIS_PATH}rating_distribution.png")
    print("Saved rating distribution plot.")

    # 2. Sentiment Mapping
    df = df[df['reviews.rating'] != 3]
    df['sentiment'] = df['reviews.rating'].apply(lambda x: 1 if x >= 4 else 0)

    # 3. Save Sample
    df.head(20).to_csv("data/eda_sample.csv", index=False)
    print("EDA Sample created.")
