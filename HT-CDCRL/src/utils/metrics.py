import torch
from torchmetrics import Metric


class Hit(Metric):
    def __init__(self, top_k):
        super().__init__()
        self.top_k = top_k
        self.add_state("hit", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        _, pred_indices = torch.topk(preds, self.top_k)
        # target should be a one-hot vector of shape [batch_size, num_classes]
        ground_truth = torch.argmax(target, -1)
        self.hit += (pred_indices == ground_truth.view(-1, 1)).sum()
        self.count += len(target)

    def compute(self):
        return self.hit / self.count if self.count > 0 else torch.tensor(0.0)


class NormalizedDCG(Metric):
    def __init__(self, top_k):
        super().__init__()
        self.top_k = top_k
        self.add_state("ndcg", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        标准NDCG@K计算
        preds: [batch_size, num_candidates] - 预测分数
        target: [batch_size, num_candidates] - 二值相关性标签 (0或1)
        """
        batch_size = preds.shape[0]
        
        # 获取top-k预测
        _, pred_indices = torch.topk(preds, min(self.top_k, preds.shape[1]), dim=1)
        
        # 计算DCG@K
        dcg_scores = torch.zeros(batch_size, device=preds.device)
        
        for i in range(batch_size):
            # 获取排序位置的相关性
            relevance = target[i][pred_indices[i]]  # [k]
            
            # 计算DCG: sum(rel_i / log2(i+2)) for i in [0, k-1]
            positions = torch.arange(1, len(relevance) + 1, device=preds.device, dtype=torch.float32)
            discount = torch.log2(positions + 1)  # log2(2), log2(3), ..., log2(k+1)
            
            dcg = (relevance / discount).sum()
            dcg_scores[i] = dcg
        
        # 计算IDCG@K (理想情况下的DCG)
        idcg_scores = torch.zeros(batch_size, device=preds.device)
        
        for i in range(batch_size):
            # 理想排序：按相关性降序排列
            ideal_relevance, _ = torch.sort(target[i], descending=True)
            ideal_relevance = ideal_relevance[:min(self.top_k, len(ideal_relevance))]
            
            # 计算IDCG
            positions = torch.arange(1, len(ideal_relevance) + 1, device=preds.device, dtype=torch.float32)
            discount = torch.log2(positions + 1)
            
            idcg = (ideal_relevance / discount).sum()
            idcg_scores[i] = idcg
        
        # NDCG = DCG / IDCG，避免除零
        ndcg_scores = torch.where(idcg_scores > 0, dcg_scores / idcg_scores, torch.zeros_like(dcg_scores))
        
        self.ndcg += ndcg_scores.sum()
        self.count += batch_size

    def compute(self):
        return self.ndcg / self.count if self.count > 0 else torch.tensor(0.0)

class Precision(Metric):
    def __init__(self, top_k):
        super().__init__()
        self.top_k = top_k
        self.add_state("precision", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        _, pred_indices = torch.topk(preds, self.top_k)
        ground_truth = torch.argmax(target, -1)
        
        # Number of hits in top-k
        hits = (pred_indices == ground_truth.view(-1, 1)).sum()
        self.precision += hits / self.top_k
        self.count += len(target)

    def compute(self):
        # The precision is already averaged over the batch in update, so we average over batches here
        return self.precision / self.count if self.count > 0 else torch.tensor(0.0)
