
import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score
from utils.fewshot_loader import FewShotDataset, PrototypicalNet, prototypical_loss

data_path = "preprocessed/test"
model_path = "/content/drive/MyDrive/brain_tumor_fewshot_checkpoints/protonet_final.pt"
csv_log_path = "/content/drive/MyDrive/brain_tumor_fewshot_checkpoints/eval_log.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_way = 4
k_shot = 7
q_queries = 10
embedding_dim = 64
num_test_episodes = 1000
log_every = 50
dataset = FewShotDataset(root_dir=data_path, image_size=256)
model = PrototypicalNet(embedding_dim=embedding_dim).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
print(f"Loaded model from: {model_path}")

#eval looop
all_preds = []
all_trues = []
acc_list = []
log_records = []

with torch.no_grad():
    for i in range(1, num_test_episodes + 1):
        support_x, support_y, query_x, query_y = dataset.sample_episode(n_way, k_shot, q_queries)

        support_x = torch.tensor(support_x).permute(0, 3, 1, 2).to(device)
        query_x = torch.tensor(query_x).permute(0, 3, 1, 2).to(device)
        support_y = torch.tensor(support_y).to(device)
        query_y = torch.tensor(query_y).to(device)

        inputs = torch.cat([support_x, query_x], dim=0)
        embeddings = model(inputs)

        loss, acc, preds = prototypical_loss(embeddings, query_y, n_way, k_shot, q_queries, return_preds=True)

        acc_list.append(acc)
        all_preds.extend(preds.cpu().numpy())
        all_trues.extend(query_y.cpu().numpy())

        if i % log_every == 0:
            avg_acc = np.mean(acc_list)
            macro_precision = precision_score(all_trues, all_preds, average='macro')
            macro_f1 = f1_score(all_trues, all_preds, average='macro')
            print(f"[Episode {i}] Acc: {avg_acc:.4f} | Precision: {macro_precision:.4f} | F1: {macro_f1:.4f}")
            log_records.append([i, avg_acc, macro_precision, macro_f1])

# csv export
df = pd.DataFrame(log_records, columns=["Episode", "Accuracy", "Precision", "F1"])
df.to_csv(csv_log_path, index=False)
print(f"Evaluation log saved to: {csv_log_path}")
#results
final_acc = np.mean(acc_list)
final_precision = precision_score(all_trues, all_preds, average='macro')
final_f1 = f1_score(all_trues, all_preds, average='macro')

print("Final Evaluation over 600 test episodes:")
print(f"Accuracy: {final_acc:.4f}")
print(f"Precision (Macro): {final_precision:.4f}")
print(f"F1 Score (Macro): {final_f1:.4f}")
