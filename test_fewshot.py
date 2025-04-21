from utils.fewshot_loader import FewShotDataset

dataset = FewShotDataset("preprocessed/train")

support_x, support_y, query_x, query_y = dataset.sample_episode(n_way=3, k_shot=5, q_queries=5)

print("Support Set Shape:", support_x.shape)
print("Query Set Shape:", query_x.shape)
print("Support Labels:", support_y)
print("Query Labels:", query_y)
