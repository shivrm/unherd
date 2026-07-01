import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import os
import requests
import zipfile

class BPRDataset(Dataset):
    def __init__(self, df, num_items, num_negatives=1):
        self.users = df['user_idx'].values
        self.positive_items = df['item_idx'].values
        self.num_items = num_items
        self.num_negatives = num_negatives

        # Create a set of (user, item) pairs for efficient lookup during negative sampling
        self.user_item_interactions = df.groupby('user_idx')['item_idx'].apply(set).to_dict()

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user_idx = self.users[idx]
        positive_item_idx = self.positive_items[idx]

        # Sample negative items
        negative_item_idx = self._sample_negative(user_idx)

        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(positive_item_idx, dtype=torch.long),
            torch.tensor(negative_item_idx, dtype=torch.long)
        )

    def _sample_negative(self, user_idx):
        user_positive_items = self.user_item_interactions.get(user_idx, set())
        while True:
            negative_item = np.random.randint(0, self.num_items)
            if negative_item not in user_positive_items:
                return negative_item
            

class MACRDataset(Dataset):
    def __init__(self, df):
        self.users = df['user_idx'].values
        self.items = df['item_idx'].values
        self.labels = df['binary_rating'].values.astype(np.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float)
        )


def balance_test_set(df, max_interactions=20):
    balanced_df_rows = []
    # Group by item_idx and sample if count exceeds max_interactions_per_item
    for item_id, group in df.groupby('item_idx'):
        if len(group) > max_interactions:
            # Randomly sample 'max_interactions_per_item' rows from the group
            balanced_df_rows.append(group.sample(n=max_interactions, random_state=42)) # Using a fixed random_state for reproducibility
        else:
            balanced_df_rows.append(group)

    # Concatenate all sampled/kept rows to form the new balanced test_df
    if balanced_df_rows:
        return pd.concat(balanced_df_rows).sort_values(by='timestamp').reset_index(drop=True)
    else:
        return pd.DataFrame(columns=df.columns) # Return empty DataFrame with original columns if no data
    
def download_movielens():
    url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
    dir = 'ml-100k'
    zip_path = 'ml-100k.zip'

 
    if not os.path.exists(dir):
        print(f"Downloading {url}.")
        r = requests.get(url)
        with open(zip_path, 'wb') as f:
            f.write(r.content)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('.')
        print("Download and extraction complete.")
    else:
        print(f"Dataset already found at {dir}.")

    return dir

def load_movielens(path, train_frac=0.8):
    ratings_path = path + '/u.data'
    ratings_df = pd.read_csv(ratings_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

    # Map user IDs to consecurive numbers
    user_ids = ratings_df['user_id'].unique().tolist()
    user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    ratings_df['user_idx'] = ratings_df['user_id'].map(user_to_idx)

    # Map item IDs to consecurive numbers
    item_ids = ratings_df['item_id'].unique().tolist()
    item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
    ratings_df['item_idx'] = ratings_df['item_id'].map(item_to_idx)

    # Binarize ratings
    ratings_df['binary_rating'] = ratings_df['rating'].apply(lambda x: 1 if x >= 4 else 0)

    # Sort by timestamp and split
    ratings_df = ratings_df.sort_values(by='timestamp').reset_index(drop=True)
    split_idx = int(len(ratings_df) * train_frac)
    train_df = ratings_df.iloc[:split_idx]
    test_df = ratings_df.iloc[split_idx:]

    return train_df, test_df