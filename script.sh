#!/bin/bash
set -e

mkdir -p data artifacts

echo "DATASET COLLECTION"
curl -L -u "$KAGGLE_USERNAME:$KAGGLE_KEY" \
    "https://www.kaggle.com/api/v1/datasets/download/ky1338/10000-movies-letterboxd-data" \
    -o data/dataset.zip

unzip -o data/dataset.zip -d ./data

CSV_FILE=$(ls data/*.csv | head -n 1)

echo "PROCESSING: $CSV_FILE, CUTOFF=$CUTOFF"

# saving header
head -n 1 "$CSV_FILE" > artifacts/processed_data.csv

# shuffling and cutting data before appending to the output file
tail -n +2 "$CSV_FILE" | shuf | head -n "$CUTOFF" >> artifacts/processed_data.csv

echo "FIN!"