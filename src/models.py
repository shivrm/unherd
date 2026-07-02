import torch
import torch.nn as nn
import torch.nn.functional as F

def bpr_loss(pos, neg):
    return -torch.sum(nn.functional.logsigmoid(pos - neg))

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embed_dim):
        super(MatrixFactorization, self).__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)

        # Initialize embeddings with a small normal distribution
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def recompute(self):
        # MF doesn't need to recompute
        pass

    def forward(self, u, i):
        e_u = self.user_emb(u)
        e_i = self.item_emb(i)
        return (e_u * e_i).sum(dim=1)

    def bce_loss(self, users, items, labels):
        pred = self.forward(users, items)
        return F.binary_cross_entropy(labels, pred)

    def bpr_loss(self, users, pos_items, neg_items):
        pos = self.forward(users, pos_items)
        neg = self.forward(users, neg_items)
        return bpr_loss(pos, neg) 


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, num_layers, adj_normalized):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.adj_normalized = adj_normalized

        self.base_user_emb = nn.Embedding(num_users, embed_dim)
        self.base_item_emb = nn.Embedding(num_items, embed_dim)

        # Initialize embeddings with a small normal distribution
        nn.init.normal_(self.base_user_emb.weight, std=0.01)
        nn.init.normal_(self.base_item_emb.weight, std=0.01)

        self.recompute()

    def recompute(self):
        embed_all = torch.cat([self.base_user_emb.weight, self.base_item_emb.weight], dim=0)
        embed_list = [embed_all]

        for _ in range(self.num_layers):
            embed_all = torch.sparse.mm(self.adj_normalized, embed_all)
            embed_list.append(embed_all)
  
        emb_all_final = torch.mean(torch.stack(embed_list), dim=0)

        self.user_emb, self.item_emb = torch.split(
            emb_all_final, [self.num_users, self.num_items]
        )

    def forward(self, u, i):
        e_u = self.user_emb[u]
        e_i = self.item_emb[i]

        return (e_u * e_i).sum(dim=1)

    def bce_loss(self, users, items, labels):
        pred = self.forward(users, items)
        return F.binary_cross_entropy(pred, labels)

    def bpr_loss(self, users, pos_items, neg_items):
        pos = self.forward(users, pos_items)
        neg = self.forward(users, neg_items)
        return bpr_loss(pos.squeeze(), neg.squeeze()) 


class MACRWrapper(nn.Module):
    def __init__(self, base_model, num_users, num_items, alpha=1e-3, beta=1e-3, c=30.0):
        super(MACRWrapper, self).__init__()
        self.base_model = base_model

        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        self.alpha = alpha  # item popularity loss weight
        self.beta = beta    # user activity loss weight
        self.c = c          # counterfactual constant

        # Initialize bias weights to zero
        nn.init.constant_(self.user_bias.weight, 0.0)
        nn.init.constant_(self.item_bias.weight, 0.0)

    def recompute(self):
        # self.base_model.recompute()
        pass

    def forward(self, user_indices, item_indices):
        # 1. Get personalized matching score from base model
        y_ui = self.base_model(user_indices, item_indices)
        if len(y_ui.shape) == 1:
            y_ui = y_ui.unsqueeze(-1)

        # 2. Compute Popularity Branches
        y_i = self.item_bias(item_indices)
        y_u = self.user_bias(user_indices)

        if self.training:
            return y_ui, y_i, y_u
        else:
            # Perform Counterfactual Inference (TIE = TE - NDE)
            sig_ui = torch.sigmoid(y_ui)
            sig_i = torch.sigmoid(y_i)
            sig_u = torch.sigmoid(y_u)

            c_val = torch.tensor(self.c).to(y_ui.device)
            score = (sig_ui - torch.sigmoid(c_val)) * sig_i * sig_u
            return score.squeeze()

    def bce_loss(self, users, items, labels):
        y_ui, y_i, y_u = self.forward(users, items)
        y_total = torch.sigmoid(y_ui) * torch.sigmoid(y_i) * torch.sigmoid(y_u)

        loss_main = F.binary_cross_entropy(y_total.squeeze(), labels)
        loss_item = F.binary_cross_entropy(torch.sigmoid(y_i).squeeze(), labels)
        loss_user = F.binary_cross_entropy(torch.sigmoid(y_u).squeeze(), labels)

        return loss_main + self.alpha * loss_item + self.beta * loss_user

    def bpr_loss(self, user_indices, pos_item_indices, neg_item_indices):
        # Calculate scores for positive samples
        y_ui_pos, y_i_pos, y_u_pos = self.forward(user_indices, pos_item_indices)
        y_total_pos = torch.sigmoid(y_ui_pos) * torch.sigmoid(y_i_pos) * torch.sigmoid(y_u_pos)

        # Calculate scores for negative samples
        y_ui_neg, y_i_neg, y_u_neg = self.forward(user_indices, neg_item_indices)
        y_total_neg = torch.sigmoid(y_ui_neg) * torch.sigmoid(y_i_neg) * torch.sigmoid(y_u_pos)

        # Standard BPR loss on combined probability
        loss_main = bpr_loss(y_total_pos.squeeze(), y_total_neg.squeeze())

        # Item popularity loss for both positive and negative items
        pos_labels = torch.ones_like(user_indices, dtype=torch.float)
        neg_labels = torch.zeros_like(pos_labels)
        loss_item = F.binary_cross_entropy(torch.sigmoid(y_i_pos).squeeze(), pos_labels) + \
                    F.binary_cross_entropy(torch.sigmoid(y_i_neg).squeeze(), neg_labels)

        # User activity loss (calculated once for the user branch)
        loss_user = F.binary_cross_entropy(torch.sigmoid(y_u_pos).squeeze(), pos_labels) + \
                    F.binary_cross_entropy(torch.sigmoid(y_u_neg).squeeze(), neg_labels)

        return loss_main + self.alpha * loss_item + self.beta * loss_user
