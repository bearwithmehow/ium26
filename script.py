import pandas as pd
import numpy as np
import kaggle
import ast
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

os.environ["KAGGLE_USERNAME"] = os.environ.get("KAGGLE_USERNAME", "")
os.environ["KAGGLE_KEY"] = os.environ.get("KAGGLE_KEY", "")
kaggle.api.dataset_download_files('ky1338/10000-movies-letterboxd-data', path='./data', unzip=True)

df = pd.read_csv("./data/Movie_Data_File.csv")
print("Dataset loaded successfully")

df = df.drop(columns=["Release_year", "Film_URL", "Owner_rating"])
df["Average_rating"] = df["Average_rating"].fillna(df["Average_rating"].mean())

df = df.dropna(subset=[
    "Genres",
    "Runtime",
    "Description"
])

df.loc[df["Runtime"] == 0, "Runtime"] = np.nan
df = df.dropna(subset=["Runtime"])
df = df.drop_duplicates()

numeric_columns = [
    "Runtime",
    "Average_rating",
    "Watches",
    "Likes",
    "Fans",
    "Total_ratings",
    "List_appearances" 
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

text_columns = [
    "Film_title",
    "Director",
    "Cast",
    "Genres",
    "Countries",
    "Description"
]

for col in text_columns:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

df["Studios"] = df["Studios"].fillna("Unknown")
df = df.dropna()

star_cols = [
    "½","★","★½","★★","★★½",
    "★★★","★★★½","★★★★","★★★★½","★★★★★"
]

for col in star_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df[col] = np.where(df["Total_ratings"] > 0, df[col] / df["Total_ratings"], 0)

scaler = MinMaxScaler()
numeric_cols_to_scale = [
    "Runtime",
    "Watches",
    "List_appearances",
    "Likes",
    "Fans",
    "Total_ratings"
]

df[numeric_cols_to_scale] = scaler.fit_transform(df[numeric_cols_to_scale])

df["Genres"] = df["Genres"].apply(ast.literal_eval)
genres = df["Genres"].explode().str.get_dummies().groupby(level=0).sum()
df = pd.concat([df, genres], axis=1)

fuzzy_genres = [
    "Captivating vision and Shakespearean drama", "Epic adventure and breathtaking battles", 
    "Epic heroes", "Epic history and literature", "Fantasy adventure, heroism, and swordplay", 
    "Historical battles and epic heroism", "Show All…", "Superheroes in action-packed battles with villains"
]
df = df.drop(columns=fuzzy_genres, errors="ignore")

train, temp = train_test_split(df, test_size=0.40, random_state=42)
dev, test = train_test_split(temp, test_size=0.50, random_state=42)

print(f"Splits generated -> Train: {len(train)}, Dev: {len(dev)}, Test: {len(test)}")

train.to_csv('./data/train.csv', index=False)
dev.to_csv('./data/dev.csv', index=False)
test.to_csv('./data/test.csv', index=False)

print("Artifacts saved to artifacts/")