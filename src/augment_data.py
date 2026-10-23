import pandas as pd
import os

def augment_dataset():
    print("Loading original dataset and local data...")
    
    # 1. Load original dataset (handling standard UCI formatting)
    df_orig = pd.read_csv('data/raw/spam.csv', encoding='latin-1')
    if 'v1' in df_orig.columns and 'v2' in df_orig.columns:
        df_orig = df_orig[['v1', 'v2']]
        df_orig.columns = ['label', 'message']
    else:
        df_orig = df_orig[['label', 'message']]

    # 2. Load Zambian local context data
    local_path = 'data/processed/local_zambian_spam.csv'
    if os.path.exists(local_path):
        df_local = pd.read_csv(local_path)
        
        # 3. Concatenate datasets
        df_combined = pd.concat([df_orig, df_local], ignore_index=True)
        
        # Shuffle to mix local patterns evenly with the base dataset
        df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save processed dataset
        os.makedirs('data/processed', exist_ok=True)
        df_combined.to_csv('data/processed/augmented_spam_dataset.csv', index=False)
        print(f"Success! Combined dataset saved. Total rows: {len(df_combined)}")
    else:
        print(f"Error: {local_path} not found. Please create the local CSV file first.")

if __name__ == "__main__":
    augment_dataset()