import torch
import math
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

class MetricsCalculator:
    @staticmethod
    def calculate_propensity(items, item_to_group, num_groups, weights=None):
        counts = np.zeros(num_groups) + 1e-9

        if weights is None:
            weights = [1.0] * len(items)

        for item, weight in zip(items, weights):
            group = item_to_group.get(item)
            if group is not None:
                counts[group] += weight

        return counts / counts.sum()

    @staticmethod
    def build_popularity_groups(num_items, popularity_list):
        sorted_items = popularity_list.sort_values(ascending=True)

        n_items = len(sorted_items)
        head_threshold = int(0.8 * n_items)
        tail_threshold = int(0.2 * n_items)

        # Map items to 0 (head), 1 (mid), 2 (tail)
        item_to_group = {}
        for i, item_idx in enumerate(sorted_items.index):
            if i < tail_threshold:
                item_to_group[item_idx] = 2
            elif i >= head_threshold:
                item_to_group[item_idx] = 0
            else:
                item_to_group[item_idx] = 1

        # Set missing items to mid
        for i in range(num_items):
            if i not in item_to_group:
                item_to_group[i] = 1 

        return item_to_group

    def __init__(self, train_df, test_df, model, device=None):
        self.train_df = train_df
        self.test_df = test_df
        self.model = model
        self.device = device

        self.train_interactions = train_df.groupby('user_idx')['item_idx'].apply(set).to_dict()
        self.item_popularity = train_df['item_idx'].value_counts().to_dict()

        # Identify long-tail items
        item_ids = set(train_df['item_id'].unique().tolist() + test_df['item_id'].unique().tolist())
        self.num_items = len(item_ids)
        item_indices = np.arange(self.num_items)
        popularity_list = pd.Series(self.item_popularity).reindex(item_indices, fill_value=0)
        items_sorted = popularity_list.sort_values(ascending=True)
        lt_threshold_idx = int(len(items_sorted) * 0.8)

        self.long_tail_indices = set(items_sorted.index[:lt_threshold_idx].tolist())

        # Map each user to their past interactions
        self.train_user_items = self.train_df.groupby("user_idx")["item_idx"].apply(list).to_dict()
        self.train_user_ratings = self.train_df.groupby("user_idx")["rating"].apply(list).to_dict()

        self.item_to_group = self.build_popularity_groups(self.num_items, popularity_list)

    def get_topk_recs(self, user_idx, k=20):
        seen_items = self.train_interactions.get(user_idx, set())

        candidate_items = [
            item for item in range(self.num_items)
            if item not in seen_items
        ]

        if not candidate_items:
            return []

        with torch.no_grad():
            user_tensor = torch.full(
                (len(candidate_items),),
                user_idx,
                dtype=torch.long,
                device=self.device,
            )

            item_tensor = torch.tensor(
                candidate_items,
                dtype=torch.long,
                device=self.device,
            )

            scores = self.model(user_tensor, item_tensor)
            top_idx = torch.argsort(scores, descending=True)[:k].cpu().numpy()

        return [candidate_items[i] for i in top_idx]

    def compute_ndcg(self, k=20):
        self.model.eval()

        test_users = self.test_df[self.test_df["binary_rating"] == 1]["user_idx"].unique()

        ndcgs = []
        for user_idx in test_users:
            positive = self.test_df[(self.test_df['user_idx'] == user_idx) & (self.test_df['binary_rating'] == 1)]

            true_items = set(positive["item_idx"].unique())
            recs = self.get_topk_recs(user_idx, k)

            if not true_items or not recs:
                continue

            dcg = 0.0
            for rank, item in enumerate(recs):
                if item in true_items:
                    dcg += 1.0 / math.log2(rank + 2)

            
            idcg = sum(
                1.0 / math.log2(rank + 2)
                for rank in range(min(k, len(true_items)))
            )

            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

        return float(np.mean(ndcgs)) if ndcgs else 0.0
    
    def compute_arp(self, k=20):
        self.model.eval()

        arps = []
        for user_idx in self.test_df["user_idx"].unique():
            recs = self.get_topk_recs(user_idx, k)
            if not recs:
                continue

            avg_popularity = np.mean([
                self.item_popularity.get(item, 0)
                for item in recs
            ])

            arps.append(avg_popularity)

        return float(np.mean(arps)) if arps else 0.0
    
    def compute_aplt(self, k=20):
        self.model.eval()

        aplts = []
        for user_idx in self.test_df["user_idx"].unique():
            recs = self.get_topk_recs(user_idx, k)
            if not recs:
                continue

            lt_count = sum(item in self.long_tail_indices for item in recs)
            tail_fraction = lt_count / len(recs)
            aplts.append(tail_fraction)

        return float(np.mean(aplts)) if aplts else 0.0
    
    def compute_aclt(self, k=20):
        self.model.eval()

        unique_tail_items = set()
        for user_idx in self.test_df["user_idx"].unique():
            recs = self.get_topk_recs(user_idx, k)
            for item in recs:
                if item in self.long_tail_indices:
                    unique_tail_items.add(item)

        if not self.long_tail_indices:
            return 0.0
        
        return len(unique_tail_items) / len(self.long_tail_indices)

    def compute_upd(self, k=20):
        self.model.eval()

        if self.item_to_group is None:
            raise ValueError("item_to_group is required for UPD.")

        upd_scores = []

        for user_idx in self.test_df["user_idx"].unique():
            hist_items = self.train_user_items.get(user_idx, [])
            hist_ratings = self.train_user_ratings.get(user_idx, [])
            if not hist_items:
                continue

            recs = self.get_topk_recs(user_idx, k)
            if not recs:
                continue

            P = self.calculate_propensity(hist_items, self.item_to_group, 3, hist_ratings)
            Q = self.calculate_propensity(recs, self.item_to_group, 3)

            js_dist = jensenshannon(P, Q)
            upd_scores.append(js_dist ** 2)

        return float(np.mean(upd_scores)) if upd_scores else 0.0
 
    def compute_all(self, k=20):
        return {
            "NDCG@K": self.compute_ndcg(k),
            "ARP@K": self.compute_arp(k),
            "APLT@K": self.compute_aplt(k),
            "ACLT@K": self.compute_aclt(k),
            "UPD@K": self.compute_upd(k)
        }

