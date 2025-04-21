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
        self.class_folders = [
            os.path.join(root_dir, cls)
            for cls in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, cls))
        ]
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
        
        # Augmentations
        if random.random() < 0.5:
            img = cv2.flip(img, 1)  # Horizontal flip
        if random.random() < 0.5:
            angle = random.randint(-15, 15)
            M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        img = img.astype(np.float32) / 255.0
        return img


    def sample_episode(self, n_way=3, k_shot=5, q_queries=5):
        selected_classes = random.sample(list(self.class_to_images.keys()), n_way)

        support_images = []
        support_labels = []

        query_images = []
        query_labels = []

        label_map = {cls: i for i, cls in enumerate(selected_classes)}

        for cls in selected_classes:
            image_paths = random.sample(self.class_to_images[cls], k_shot + q_queries)
            support = image_paths[:k_shot]
            query = image_paths[k_shot:]

            for img_path in support:
                img = self._load_image(img_path)
                support_images.append(img)
                support_labels.append(label_map[cls])

            for img_path in query:
                img = self._load_image(img_path)
                query_images.append(img)
                query_labels.append(label_map[cls])

        return (
            np.array(support_images), np.array(support_labels),
            np.array(query_images), np.array(query_labels)
        )

class PrototypicalNet(nn.Module):
    def __init__(self, input_shape=(3, 256, 256), embedding_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout(0.2), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(0.2), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(0.2), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )

        

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

def prototypical_loss(embeddings, targets, n_way, k_shot, q_queries):
    support = embeddings[:n_way * k_shot]
    query = embeddings[n_way * k_shot:]
    
    support = F.normalize(support, dim=1)
    query = F.normalize(query, dim=1)
    # Compute prototypes
    prototypes = []
    for i in range(n_way):
        class_embeddings = support[i * k_shot : (i + 1) * k_shot]
        prototype = class_embeddings.mean(0)
        prototypes.append(prototype)
    prototypes = torch.stack(prototypes)

    # Compute distances
    dists = torch.cdist(query, prototypes)  # (num_queries, num_classes)
    labels = torch.arange(n_way).unsqueeze(1).expand(n_way, q_queries).reshape(-1).to(dists.device)
    loss = F.cross_entropy(-dists, labels)
    preds = dists.argmin(dim=1)
    acc = (preds == labels).float().mean().item()
    return loss, acc
