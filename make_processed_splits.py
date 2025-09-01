import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Path to your interim cleaned dataset
input_path = "data/interim/smart_logistics_cleaned.csv"

# Output folder
output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

# Load the cleaned dataset
df = pd.read_csv(input_path)

# Split into train (70%), validation (15%), test (15%)
train_df, temp_df = train_test_split(
    df, test_size=0.3, random_state=42, stratify=df["logistics_delay"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df["logistics_delay"]
)

# Save to processed folder
train_df.to_csv(f"{output_dir}/train.csv", index=False)
val_df.to_csv(f"{output_dir}/val.csv", index=False)
test_df.to_csv(f"{output_dir}/test.csv", index=False)

print("✅ Split completed. Files saved in data/processed/")
