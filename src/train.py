import dataset
import metrics
import models
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

device = 'cpu'

def train_bpr(model, train_df, test_df, epochs=3):
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    calc = metrics.MetricsCalculator(train_df, test_df, model, device)

    train_ds = dataset.BPRDataset(train_df, num_items)
    train_loader = DataLoader(train_ds, batch_size=32)

    for epoch in range(epochs):
        progress = tqdm(train_loader, f"Epoch {epoch + 1}/{epochs}")

        for u, pos, neg in progress:
            model.train()
            optim.zero_grad()
            loss = model.bpr_loss(u.to(device), pos.to(device), neg.to(device))
            loss.backward()
            optim.step()
            model.recompute()

            ndcg = calc.compute_ndcg()

            desc = f'Loss: {float(loss.item()):.4f}, NDCG: {ndcg:.4f}'
            progress.set_postfix_str(desc)
        
        m = calc.compute_all()
        s = ', '.join(f'{k}: {v:.4f}' for k, v in m.items())
        print("Metrics:", s)
     
def train_bce(model, train_df, test_df, epochs=3):
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    calc = metrics.MetricsCalculator(train_df, test_df, model, device)

    train_ds = dataset.BCEDataset(train_df)
    train_loader = DataLoader(train_ds, batch_size=32)

    for epoch in range(epochs):
        progress = tqdm(train_loader, f"Epoch {epoch + 1}/{epochs}")

        for user, item, label in progress:
            model.train()
            optim.zero_grad()
            loss = model.bce_loss(user.to(device), item.to(device), label.to(device))
            loss.backward()
            optim.step()
            model.recompute()

            ndcg = calc.compute_ndcg()

            desc = f'Loss: {float(loss.item()):.4f}, NDCG: {ndcg:.4f}'
            progress.set_postfix_str(desc)
        
        m = calc.compute_all()
        s = ', '.join(f'{k}: {v:.4f}' for k, v in m.items())
        print("Metrics:", s)

if __name__ == '__main__':
    path = dataset.download_movielens()
    num_users, num_items, train_df, test_df = dataset.load_movielens(path)
    test_df = dataset.balance_test_set(test_df)

    # Train and save MF Models
    mf_base_64 = models.MatrixFactorization(num_users, num_items, 64).to(device)
    train_bpr(mf_base_64, train_df, test_df, epochs=2)
    torch.save(mf_base_64, 'models/mf-base-64.pt')

    for param in mf_base_64.parameters():
        param.requires_grad = False

    mf_macr_64 = models.MACRWrapper(mf_base_64, num_users, num_items)
    train_bce(mf_macr_64, train_df, test_df, epochs=1)
    torch.save(mf_macr_64, 'models/mf-macr-64.pt')


    # Train and save LightGCN Models
    adj_normalized = dataset.compute_normalized_adj(train_df, num_users, num_items)    

    gcn_base_64 = models.LightGCN(num_users, num_items, 64, 3, adj_normalized)
    train_bpr(gcn_base_64, train_df, test_df, epochs=1)
    torch.save(gcn_base_64, 'models/gcn-base-64.pt')

    for param in gcn_base_64.parameters():
        param.requires_grad = False

    gcn_macr_64 = models.MACRWrapper(gcn_base_64, num_users, num_items)
    train_bce(gcn_macr_64, train_df, test_df, epochs=1)
    torch.save(gcn_macr_64, 'models/gcn-macr-64.pt')



    
