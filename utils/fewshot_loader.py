#structure for fewshot training
import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

class FewShotDataset:
    def __init__(self, root_dir, image_size=256):
        self.root_dir = root_dir
        self.image_size = image_size
        self.class_to_images = {
            cls: self._glob_images(os.path.join(root_dir, cls))
            for cls in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, cls))
        }

    def _glob_images(self, class_dir):
        return [
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

    def _load_image(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0
        return img

    #defining fewshot alg
    def sample_episode(self, n_way=3, k_shot=5, q_queries=5):
        selected_classes = random.sample(list(self.class_to_images.keys()), n_way)
        support_images, support_labels = [], []
        query_images, query_labels = [], []
        label_map = {cls: i for i, cls in enumerate(selected_classes)}

        for cls in selected_classes:
            images = random.sample(self.class_to_images[cls], k_shot + q_queries)
            support = images[:k_shot]
            query = images[k_shot:]

            for img_path in support:
                support_images.append(self._load_image(img_path))
                support_labels.append(label_map[cls])

            for img_path in query:
                query_images.append(self._load_image(img_path))
                query_labels.append(label_map[cls])

        return (
            np.array(support_images),
            np.array(support_labels),
            np.array(query_images),
            np.array(query_labels),
        )
# concept of prototypical net - Prototypical Networks learn to map images into an embedding space 
# where each class is represented by the mean of its support embeddings (a prototype). During inference, a query image is classified by 
# finding the nearest prototype using a distance metric like Euclidean distance
class PrototypicalNet(nn.Module):
    def __init__(self, input_shape=(3, 256, 256), embedding_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, embedding_dim, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

def prototypical_loss(embeddings, targets, n_way, k_shot, q_queries):
    support = embeddings[:n_way * k_shot]
    query = embeddings[n_way * k_shot:]
    prototypes = []
    for i in range(n_way):
        class_embeddings = support[i * k_shot : (i + 1) * k_shot]
        prototype = class_embeddings.mean(0)
        prototypes.append(prototype)
    prototypes = torch.stack(prototypes)
    dists = torch.cdist(query, prototypes)
    labels = torch.arange(n_way).unsqueeze(1).expand(n_way, q_queries).reshape(-1).to(dists.device)
    loss = F.cross_entropy(-dists, labels)
    preds = dists.argmin(dim=1)
    acc = (preds == labels).float().mean().item()
    return loss, acc
