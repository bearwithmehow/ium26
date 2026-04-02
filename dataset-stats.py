import pandas as pd
import os

if not os.path.exists('./artifacts/train.csv'):
    print("ERR: No generated datasets found.")
    exit(1)

train = pd.read_csv('./artifacts/train.csv')
dev = pd.read_csv('./artifacts/dev.csv')
test = pd.read_csv('./artifacts/test.csv')

print("DATASET STATS")
print("=========")
print(f"Number of rows - Train: {len(train)}")
print(f"Number of rows - Dev:   {len(dev)}")
print(f"Number of rows - Test:  {len(test)}")

print(f"\n[Columns]  ({len(train.columns)} total)")
print("  " + ", ".join(train.columns.tolist()))

print("\nAverage rating ")
if 'Average_rating' in train.columns:
    print(train['Average_rating'].describe())
else:
    print("No average rating column found in the dataset")

print("\nMissing values")
missing_values = train.isnull().sum()
missing_filtered = missing_values[missing_values > 0]
if not missing_filtered.empty:
    print(missing_filtered)
else:
    print("No missing values found")

print(f"\n[Sample rows — train.head(3)]")
print(train.head(3).to_string())

