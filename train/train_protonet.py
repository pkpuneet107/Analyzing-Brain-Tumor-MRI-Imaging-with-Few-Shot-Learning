import os
import sys
import time
import torch
import numpy as np

# Make sure the project root is in the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.fewshot_loader import FewShotDataset, PrototypicalNet, prototypical_loss

# === CONFIGURATION ===
data_path = "preprocessed/train"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_way = 3             # Classes per episode
k_shot = 7           # Support examples per class
q_queries = 10        # Query examples per class
embedding_dim = 64    # Output embedding size
num_episodes = 5000   # Total training episodes
print_every = 50      # Logging interval

dataset = FewShotDataset(data_path)

model = PrototypicalNet(embedding_dim=embedding_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# === TRAINING LOOP ===
print("🚀 Starting training...")
start = time.time()

for episode in range(1, num_episodes + 1):
    support_x, support_y, query_x, query_y = dataset.sample_episode(n_way, k_shot, q_queries)

    # Combine all images
    x = np.concatenate([support_x, query_x], axis=0)
    x = torch.tensor(x.transpose(0, 3, 1, 2)).float().to(device)  # Convert to (B, C, H, W)

    model.train()
    embeddings = model(x)
    loss, acc = prototypical_loss(embeddings, None, n_way, k_shot, q_queries)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if episode % print_every == 0:
        elapsed = time.time() - start
        print(f"[Episode {episode}] Loss: {loss.item():.4f} | Accuracy: {acc*100:.2f}% | Elapsed: {elapsed:.1f}s")
        start = time.time()

print("✅ Training complete!")
