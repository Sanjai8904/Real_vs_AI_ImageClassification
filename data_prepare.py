
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.feature_extraction import extract_features


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "dataset" / "cifake"

OUTPUT_PATH = BASE_DIR / "dataset" / "image_features.csv"


# Our project label mapping
label_mapping = {
    "real": 0,
    "ai_generated": 1
}


records = []

# Process training and testing images separately
for split in ["train", "test"]:

    print(f"\nProcessing {split} images...")

    for class_name, label in label_mapping.items():

        folder = DATA_DIR / split / class_name

        image_paths = sorted(folder.glob("*.png"))

        print(f"\nExtracting features: {class_name}")

        for image_path in tqdm(image_paths):

            features = extract_features(image_path)

            features["image_name"] = image_path.name
            features["split"] = split
            features["label"] = label

            records.append(features)


# Convert extracted features into a DataFrame
df = pd.DataFrame(records)

# Save the feature dataset
df.to_csv(OUTPUT_PATH, index=False)


# Display dataset information
print("\n========== FEATURE DATASET SUMMARY ==========")

print("Dataset shape:", df.shape)

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nSplit distribution:")
print(df["split"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

print("\nFeature dataset saved to:", OUTPUT_PATH)
