import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.fewshot_loader import FewShotDataset, PrototypicalNet, prototypical_loss

data_path = "preprocessed/train"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_way = 4             # All 4 classes, OVERWROTE OG PARAMETERS
k_shot = 7            # Support examples per class
q_queries = 10        # Query examples per class
embedding_dim = 64
num_episodes = 5000
print_every = 50
log_path = "/content/drive/MyDrive/brain_tumor_fewshot_checkpoints/training_log.csv" # save to my puneetypie drive
model_path = "/content/drive/MyDrive/brain_tumor_fewshot_checkpoints/protonet_final.pt"

dataset = FewShotDataset(root_dir=data_path, image_size=256)
model = PrototypicalNet(embedding_dim=embedding_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
episode_log = []

#loop for training
print("Starting training...")
for episode in tqdm(range(1, num_episodes + 1)):
    support_x, support_y, query_x, query_y = dataset.sample_episode(n_way, k_shot, q_queries)

    support_x = torch.tensor(support_x).permute(0, 3, 1, 2).to(device)
    query_x = torch.tensor(query_x).permute(0, 3, 1, 2).to(device)
    support_y = torch.tensor(support_y).to(device)
    query_y = torch.tensor(query_y).to(device)

    inputs = torch.cat([support_x, query_x], dim=0)
    embeddings = model(inputs)

    loss, acc = prototypical_loss(embeddings, query_y, n_way, k_shot, q_queries)
    episode_log.append((episode, loss.item(), acc))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if episode % print_every == 0:
        print(f"[Episode {episode}] Loss: {loss.item():.4f} | Accuracy: {acc * 100:.2f}%")

df = pd.DataFrame(episode_log, columns=["episode", "loss", "accuracy"])
df.to_csv(log_path, index=False)
print(f"Training log saved to: {log_path}")
torch.save(model.state_dict(), model_path)
print(f"Model saved to: {model_path}")
