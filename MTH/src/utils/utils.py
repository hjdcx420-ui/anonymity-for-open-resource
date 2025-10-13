import torch
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader

class RunningMeanStd:
    # `shape` is the shape of the data to be normalized, e.g. (), (H, W), ...
    def __init__(self, shape=()):
        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.var = torch.ones(shape, dtype=torch.float32)
        self.count = 1e-4

    def update(self, x: torch.Tensor) -> None:
        """
        Update the mean and var with a new batch of data.
        x: a tensor of shape (batch_size, *shape)
        """
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

class NegotiationEnv:
    """
    模拟用户和商户之间交互的强化学习环境。
    """
    def __init__(self, merchants_embedding, distance_matrix, alpha=1.0, beta=0.1, gamma=0.1):
        """
        初始化环境.
        Args:
            merchants_embedding: 商家(POI)的嵌入层.
            distance_matrix: POI之间的地理距离矩阵.
            alpha (float): 历史一致性奖励权重.
            beta (float): 地理位置奖励权重.
            gamma (float): 多样性奖励权重.
        """
        self.merchants = merchants_embedding
        self.distance_matrix = distance_matrix.to(merchants_embedding.weight.device)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def reset_user_merchants(self, merchants_embeddings):
        self.merchants = merchants_embeddings

    def step(self, actions_users_probs, selected_merchants_indices, actions_merchant_probs, obs, ground_truth_onehot=None, eps=1e-8):
        """
        KDD'23 Inspired Reward Shaping: 引入地理和多样性奖励
        """
        chosen_idx_in_candidates = torch.argmax(actions_users_probs, dim=-1)
        chosen_poi_global_index = selected_merchants_indices[chosen_idx_in_candidates]

        # 1. 历史一致性奖励 (r_consistency) - 代理的任务奖励
        user_history_pois = obs[0].long().flatten()
        # Convert 1-based POI indices to 0-based indices
        user_history_pois_0based = user_history_pois - 1
        user_history_pois_0based = torch.clamp(user_history_pois_0based, 0, self.merchants.num_embeddings - 1)
        last_poi = user_history_pois_0based[-1]
        
        # Convert chosen_poi_global_index to 0-based for merchants embedding
        chosen_poi_for_embedding = chosen_poi_global_index - 1
        chosen_poi_for_embedding = torch.clamp(chosen_poi_for_embedding, 0, self.merchants.num_embeddings - 1)
        chosen_poi_embedding = self.merchants(chosen_poi_for_embedding)
        history_embeddings = self.merchants(user_history_pois_0based)
        
        similarity_to_history = F.cosine_similarity(
            chosen_poi_embedding,
            history_embeddings.mean(dim=0, keepdim=True)
        ).mean()
        r_consistency = torch.clamp(similarity_to_history, -1.0, 1.0)

        # 2. 地理感知奖励 (r_geo) - 惩罚不合理的地理跳跃
        # Convert chosen_poi_global_index to 0-based if needed
        chosen_poi_0based = chosen_poi_global_index - 1
        chosen_poi_0based = torch.clamp(chosen_poi_0based, 0, self.distance_matrix.shape[0] - 1)
        distance = self.distance_matrix[last_poi, chosen_poi_0based]
        # 使用指数衰减函数作为奖励，距离越近奖励越高，超过阈值则惩罚
        r_geo = torch.exp(-distance / 10.0) * 2 - 1 # 奖励范围 [-1, 1], 10km作为尺度
        
        # 3. 多样性奖励 (r_diversity) - 鼓励探索
        # Convert selected_merchants_indices to 0-based for merchants embedding
        selected_merchants_0based = selected_merchants_indices - 1
        selected_merchants_0based = torch.clamp(selected_merchants_0based, 0, self.merchants.num_embeddings - 1)
        candidate_embeddings = self.merchants(selected_merchants_0based)
        # 计算选择的POI与候选集中其他POI的平均不相似度
        dissimilarity = 1 - F.cosine_similarity(chosen_poi_embedding, candidate_embeddings, dim=-1)
        r_diversity = dissimilarity.mean()

        # 4. 最终用户奖励
        user_reward = (
            self.alpha * r_consistency +
            self.beta * r_geo +
            self.gamma * r_diversity
        )

        # 5. 商家奖励
        incentive_for_choice = actions_merchant_probs.squeeze(-1)[chosen_idx_in_candidates]
        # 商家奖励与用户满意度（一致性）和地理合理性挂钩
        merchant_reward = (r_consistency + r_geo) - incentive_for_choice * 0.5
        
        return None, user_reward.detach(), merchant_reward.detach()


class CommitteeElection:
    """
    根据用户历史、时间、地理等多种因素，为用户推荐候选商户（POI）。
    这是核心的负采样/候选集生成模块。
    """
    def __init__(self, data_dir, poi_long_term, beta=0.7, candi_merchant_num=10, distance_metrix=None, seed=42):
        self.merchant = poi_long_term
        self.distance_metrix = distance_metrix
        if self.distance_metrix is not None:
            self.num_poi = self.distance_metrix.shape[0]
        else:
            raise ValueError("distance_metrix is None, cannot determine POI number")

        day_tensors_path = os.path.join(data_dir, 'day_tensors.pt')
        if os.path.exists(day_tensors_path):
            self.day_tensors = torch.load(day_tensors_path)
        else:
            self.day_tensors = torch.zeros((self.num_poi, 7), dtype=torch.float32)
            for poi, data in self.merchant.items():
                for day in data['week']:
                    # Convert 1-based POI index to 0-based index
                    poi_idx = poi - 1
                    if 0 <= poi_idx < self.num_poi:
                        self.day_tensors[poi_idx][day] += 1.0
            self.day_tensors += 0.01 # Smoothing
            self.day_tensors = self.day_tensors / self.day_tensors.sum(dim=1, keepdim=True)
            torch.save(self.day_tensors, day_tensors_path)

        hour_tensors_path = os.path.join(data_dir, 'hour_tensors.pt')
        if os.path.exists(hour_tensors_path):
            self.hour_tensors = torch.load(hour_tensors_path)
        else:
            self.hour_tensor= torch.zeros((self.num_poi, 24), dtype=torch.float32)
            for poi, data in self.merchant.items():
                for hour in data['hour']:
                    # Convert 1-based POI index to 0-based index
                    poi_idx = poi - 1
                    if 0 <= poi_idx < self.num_poi:
                        self.hour_tensors[poi_idx][hour] += 1.0
            self.hour_tensors += 0.01 # Smoothing
            self.hour_tensors = self.hour_tensors / self.hour_tensors.sum(dim=1, keepdim=True)
            torch.save(self.hour_tensors, hour_tensors_path)

        self.beta = beta
        self.candi_merchant_num = candi_merchant_num
        self.seed = seed

    def run_election(self, poi_embeddings, obs, user_embedding):
        # 支持 batch_size > 1
        batch_size = user_embedding.shape[0] if user_embedding.dim() > 1 else 1
        poi_seq = obs[0].long()
        days = obs[3].long()
        hours = obs[4].long()

        # 1. 用户与 POI 的相似度
        user_embedding_norm = F.normalize(user_embedding, dim=-1)  # [B, D]
        poi_embeddings_norm = F.normalize(poi_embeddings, dim=-1)  # [N, D]
        user_match_score = torch.matmul(user_embedding_norm, poi_embeddings_norm.T)  # [B, N]

        # 2. 用户历史兴趣与 POI 的匹配
        # 对每个 batch 计算历史兴趣
        # Convert 1-based POI indices to 0-based indices
        poi_seq_0based = poi_seq - 1
        # Ensure indices are within bounds
        poi_seq_0based = torch.clamp(poi_seq_0based, 0, poi_embeddings.shape[0] - 1)
        selected_poi_embeddings = poi_embeddings[poi_seq_0based]  # [B, S, D]
        ave_poi_embedding = selected_poi_embeddings.mean(dim=1)  # [B, D]
        poi_match_score = F.cosine_similarity(ave_poi_embedding.unsqueeze(1), poi_embeddings.unsqueeze(0), dim=-1)  # [B, N]

        # 3. 时间分布加权：日
        day_weights = []
        for i in range(batch_size):
            day_tensor = torch.bincount(days[i], minlength=7).float().to(poi_embeddings.device)
            day_tensor = day_tensor / (day_tensor.sum() + 1e-8)
            # 确保 day_tensors 形状正确
            day_tensors_corrected = self.day_tensors.to(day_tensor.device)
            if day_tensors_corrected.shape[0] != poi_embeddings.shape[0]:
                # 如果形状不匹配，调整到正确的POI数量
                day_tensors_corrected = day_tensors_corrected[:poi_embeddings.shape[0]]
            day_weights.append(day_tensors_corrected @ day_tensor)  # [N]
        day_weights = torch.stack(day_weights, dim=0)  # [B, N]

        # 4. 时间分布加权：时
        hour_weights = []
        for i in range(batch_size):
            hour_tensor = torch.bincount(hours[i], minlength=24).float().to(poi_embeddings.device)
            hour_tensor = hour_tensor / (hour_tensor.sum() + 1e-8)
            # 确保 hour_tensors 形状正确
            hour_tensors_corrected = self.hour_tensors.to(hour_tensor.device)
            if hour_tensors_corrected.shape[0] != poi_embeddings.shape[0]:
                # 如果形状不匹配，调整到正确的POI数量
                hour_tensors_corrected = hour_tensors_corrected[:poi_embeddings.shape[0]]
            hour_weights.append(hour_tensors_corrected @ hour_tensor)  # [N]
        hour_weights = torch.stack(hour_weights, dim=0)  # [B, N]

        # 5. 历史序列衰减权重
        poi_weights = []
        for i in range(batch_size):
            T = len(poi_seq[i])
            gamma = 0.8
            decay_weights = gamma ** torch.arange(T - 1, -1, -1, device=poi_seq.device).float()
            decay_weights = decay_weights / (decay_weights.sum() + 1e-8)
            pw = torch.zeros(self.num_poi, dtype=torch.float32, device=poi_seq.device)
            # Convert 1-based POI indices to 0-based indices
            poi_seq_0based = poi_seq[i] - 1
            # Ensure indices are within bounds
            poi_seq_0based = torch.clamp(poi_seq_0based, 0, self.num_poi - 1)
            pw.index_add_(0, poi_seq_0based, decay_weights)
            poi_weights.append(pw)
        poi_weights = torch.stack(poi_weights, dim=0)  # [B, N]

        # 6. 距离加权
        distance_weight = []
        for i in range(batch_size):
            latest_poi = poi_seq[i][-1]
            # Convert 1-based POI index to 0-based index
            latest_poi_0based = latest_poi - 1
            # Ensure index is within bounds
            latest_poi_0based = torch.clamp(latest_poi_0based, 0, self.num_poi - 1)
            distance_vec = self.distance_metrix[latest_poi_0based].to(poi_seq.device)
            distance_weight.append(distance_vec / (distance_vec.max() + 1e-8))
        distance_weight = torch.stack(distance_weight, dim=0)  # [B, N]

        # 最终得分
        scores = poi_match_score + user_match_score + poi_weights + hour_weights + day_weights + distance_weight  # [B, N]


        # 困难负样本策略
        total_candidates = self.candi_merchant_num

        # 1. 获取最相似的候选 (20% - 容易正样本)
        easy_num = max(1, int(total_candidates * 0.2))
        _, top_indices = torch.topk(scores, k=min(easy_num * 3, scores.shape[1]), dim=1)  # 取3倍候选
        easy_indices = top_indices[:, :easy_num]

        # 2. 困难负样本 (60% - 中等相似度，容易混淆)
        hard_num = int(total_candidates * 0.6)
        # 选择相似度在中等范围的POI作为困难负样本
        percentile_70 = torch.quantile(scores, 0.7, dim=1, keepdim=True)
        percentile_30 = torch.quantile(scores, 0.3, dim=1, keepdim=True)

        hard_candidates_mask = (scores >= percentile_30) & (scores <= percentile_70)
        # 排除已选择的容易候选
        for i in range(batch_size):
            hard_candidates_mask[i, easy_indices[i]] = False

        hard_indices_list = []
        for i in range(batch_size):
            available_hard = hard_candidates_mask[i].nonzero(as_tuple=False).squeeze(-1)
            if len(available_hard) >= hard_num:
                # 在困难候选中随机选择，使用种子
                # 修复设备不匹配问题：Generator必须在CPU上创建
                generator = torch.Generator(device='cpu')
                generator.manual_seed(self.seed + i)  # 为每个batch使用不同的种子
                selected_hard = available_hard[torch.randperm(len(available_hard), generator=generator)[:hard_num]]
            else:
                # 如果困难候选不足，用随机候选补充
                remaining_mask = torch.ones(scores.shape[1], dtype=torch.bool, device=scores.device)
                remaining_mask[easy_indices[i]] = False
                if len(available_hard) > 0:
                    remaining_mask[available_hard] = False
                remaining = remaining_mask.nonzero(as_tuple=False).squeeze(-1)
                
                if len(remaining) > 0:
                    # 修复设备不匹配问题：Generator必须在CPU上创建
                    generator = torch.Generator(device='cpu')
                    generator.manual_seed(self.seed + i + 1000)  # 使用不同的种子偏移
                    supplement = remaining[torch.randperm(len(remaining), generator=generator)[:hard_num - len(available_hard)]]
                    selected_hard = torch.cat([available_hard, supplement])
                else:
                    selected_hard = available_hard
            
            hard_indices_list.append(selected_hard)

        # 统一hard_indices的长度
        max_hard_len = max(len(idx) for idx in hard_indices_list)
        hard_indices = []
        for idx in hard_indices_list:
            if len(idx) < max_hard_len:
                # 用easy_indices补充
                supplement_needed = max_hard_len - len(idx)
                supplement = easy_indices[len(hard_indices)][:supplement_needed]
                padded = torch.cat([idx, supplement])
            else:
                padded = idx[:max_hard_len]
            hard_indices.append(padded)

        hard_indices = torch.stack(hard_indices)[:, :hard_num]

        # 3. 随机负样本 (20% - 完全随机)
        random_num = total_candidates - easy_num - hard_num
        if random_num > 0:
            random_indices_list = []
            for i in range(batch_size):
                excluded = torch.cat([easy_indices[i], hard_indices[i]])
                remaining_mask = torch.ones(scores.shape[1], dtype=torch.bool, device=scores.device)
                remaining_mask[excluded] = False
                remaining = remaining_mask.nonzero(as_tuple=False).squeeze(-1)
                
                if len(remaining) >= random_num:
                    # 修复设备不匹配问题：Generator必须在CPU上创建
                    generator = torch.Generator(device='cpu')
                    generator.manual_seed(self.seed + i + 2000)  # 使用不同的种子偏移
                    selected_random = remaining[torch.randperm(len(remaining), generator=generator)[:random_num]]
                else:
                    selected_random = remaining
                random_indices_list.append(selected_random)
            
            # 统一random_indices的长度
            max_random_len = max(len(idx) for idx in random_indices_list)
            random_indices = []
            for idx in random_indices_list:
                if len(idx) < max_random_len:
                    supplement_needed = max_random_len - len(idx)
                    supplement = easy_indices[len(random_indices)][:supplement_needed]
                    padded = torch.cat([idx, supplement])
                else:
                    padded = idx[:max_random_len]
                random_indices.append(padded)
            
            random_indices = torch.stack(random_indices)[:, :random_num]
            
            # 组合所有候选
            all_indices = torch.cat([easy_indices, hard_indices, random_indices], dim=1)
        else:
            all_indices = torch.cat([easy_indices, hard_indices], dim=1)
    
        return all_indices

class RolloutBuffer:
    """
    用于存储强化学习中的经验数据。
    """
    def __init__(self, data_dir):
        self.data = {'user': [], 'merchant': []}
        self.seq_data = []
        self.data_dir = data_dir

    def store(self, idx, obs, act, other_act, rew, next_obs, user_embedding, merchants_embeddings, selected_action, agent_type):
        num = idx // 5000
        while len(self.seq_data) <= num:
            self.seq_data.append({'user': [], 'merchant': []})
        self.seq_data[num][agent_type].append((obs, act, other_act, rew, next_obs, user_embedding, merchants_embeddings, selected_action))

    def sample(self, agent_type):
        obs, act, other_act, rew, next_obs, user_embedding, merchants_embeddings, selected_action = zip(*self.data[agent_type])
        return obs, act, other_act, rew, next_obs, user_embedding, merchants_embeddings, selected_action

    def clear(self):
        self.data = {'user': [], 'merchant': []}
        self.seq_data = []

    def merge_data(self):
        for seq in self.seq_data:
            self.data['user'].extend(seq['user'])
            self.data['merchant'].extend(seq['merchant'])
        self.seq_data.clear()

    def __len__(self):
        # 错误：之前这里返回的是 len(self.data['user'])，它只在 merge_data() 后才有值。
        # 正确做法：应该返回在训练过程中实际存入 seq_data 的数据量。
        return sum(len(d['user']) for d in self.seq_data)


class RolloutDataset(Dataset):
    """
    将经验数据转换为PyTorch Dataset。
    """
    def __init__(self, obs, act, other_act, rew, next_obs, user_embedding, merchants_embeddings, selected_action):
        self.obs = list(obs)
        self.act = torch.stack(act)
        self.other_act = torch.stack(other_act)
        self.rew = torch.stack(rew)
        self.next_obs = list(next_obs)
        self.user_embedding = torch.stack(user_embedding)
        self.merchants_embeddings = torch.stack(merchants_embeddings)
        self.selected_action = torch.stack(selected_action)

    def __len__(self):
        return len(self.act)

    def __getitem__(self, idx):
        return self.obs[idx], self.act[idx], self.other_act[idx], self.rew[idx], self.next_obs[idx], self.user_embedding[idx], self.merchants_embeddings[idx], self.selected_action[idx]


def custom_collate_fn(batch):
    obs_list, act_list, other_act_list, rew_list, next_obs_list, user_embedding_list, merchants_embeddings_list, selected_action = zip(*batch)
    return (
        list(obs_list),
        torch.stack(act_list),
        torch.stack(other_act_list),
        torch.stack(rew_list),
        list(next_obs_list),
        torch.stack(user_embedding_list),
        torch.stack(merchants_embeddings_list),
        torch.stack(selected_action)
    )

def create_dataloader_from_buffer(buffer, agent_type, batch_size=128):
    obs, act, other_act, rew, next_obs, user_embedding, merchants_embeddings, selected_action = buffer.sample(agent_type)
    if agent_type != 'user':
        # Ensure 'act' for merchant is a tensor
        act = [torch.tensor(i) if not isinstance(i, torch.Tensor) else i for i in act]
    dataset = RolloutDataset(obs, act, other_act, rew, next_obs, user_embedding, merchants_embeddings, selected_action)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

class ImprovedNegotiationEnv:
    def __init__(self, merchants_embedding, alpha=2.0, beta=0.5, gamma=0.3, delta=0.2):
        self.merchants = merchants_embedding
        self.alpha = alpha    # 增加任务奖励权重
        self.beta = beta      # 增加相似性奖励权重
        self.gamma = gamma    # 激励奖励权重
        self.delta = delta    # 排序奖励权重（新增）
        
    def step(self, actions_users_probs, selected_merchants_indices, actions_merchant_probs, obs, ground_truth_onehot, eps=1e-8):
        """
        重新设计的奖励函数，与Hit@K指标强对齐
        """
        chosen_idx_in_candidates = torch.argmax(actions_users_probs, dim=-1)
        chosen_poi_global_index = selected_merchants_indices[chosen_idx_in_candidates]
        gt_poi_global_index = torch.argmax(ground_truth_onehot, dim=-1)
        is_correct = (chosen_poi_global_index == gt_poi_global_index).float()
        
        # 1. 强化Hit@1奖励 (直接对应Hit@1指标)
        r_hit1 = is_correct * 5.0 + (1 - is_correct) * -1.0
        
        # 2. Ranking质量奖励 (对应Hit@K指标)
        # 计算ground truth在概率排序中的位置
        sorted_probs, sorted_indices = torch.sort(actions_users_probs, descending=True)
        
        # 找到ground truth的排序位置
        gt_mask = (selected_merchants_indices == gt_poi_global_index)
        if gt_mask.any():
            gt_candidate_idx = gt_mask.nonzero(as_tuple=False)[0, 0]
            gt_rank = (sorted_indices == gt_candidate_idx).nonzero(as_tuple=False)[0, 0] + 1  # 1-based rank
            
            # 基于排序位置的奖励 (越靠前奖励越高)
            r_ranking = torch.clamp(10.0 / gt_rank.float(), 0.0, 10.0)  # rank=1得10分，rank=10得1分
            
            # Hit@K奖励
            r_hit5 = 1.0 if gt_rank <= 5 else -0.2
            r_hit10 = 0.5 if gt_rank <= 10 else -0.1
        else:
            r_ranking = -2.0  # ground truth不在候选集中，重罚
            r_hit5 = -0.5
            r_hit10 = -0.2
        
        # 3. 概率分布质量奖励
        gt_prob = actions_users_probs[chosen_idx_in_candidates] if is_correct else 0.0
        max_wrong_prob = torch.max(actions_users_probs * (1 - gt_mask.float())) if gt_mask.any() else actions_users_probs.max()
        r_prob_quality = torch.clamp(gt_prob - max_wrong_prob, -1.0, 1.0)
        
        # 4. 多样性探索奖励 (防止模型总是选择相同的POI)
        entropy = -torch.sum(actions_users_probs * torch.log(actions_users_probs + eps))
        r_exploration = torch.clamp(entropy / 2.0, 0.0, 1.0)  # 归一化熵奖励
        
        # 5. 组合奖励 (权重设计与评估指标对齐)
        user_reward = (
            self.alpha * r_hit1 +           # 5.0 - Hit@1对齐
            0.8 * r_ranking +               # 排序质量
            0.6 * r_hit5 +                  # Hit@5对齐  
            0.4 * r_hit10 +                 # Hit@10对齐
            self.beta * r_prob_quality +    # 概率质量
            0.2 * r_exploration             # 探索奖励
        )
        
        # 商家奖励
        incentive_for_choice = actions_merchant_probs.squeeze(-1)[chosen_idx_in_candidates]
        merchant_reward = is_correct * 2.0 - incentive_for_choice * 0.5
        
        # 确保输出是标量
        if user_reward.dim() > 0:
            user_reward = user_reward.mean()
        if merchant_reward.dim() > 0:
            merchant_reward = merchant_reward.mean()
        
        return None, user_reward.detach(), merchant_reward.detach()

class ImprovedCommitteeElection:
    def __init__(self, data_dir, poi_long_term, beta=0.7, candi_merchant_num=10, 
                 distance_metrix=None, hard_negative_ratio=0.3):
        self.merchant = poi_long_term
        self.distance_metrix = distance_metrix
        if self.distance_metrix is not None:
            self.num_poi = self.distance_metrix.shape[0]
        else:
            raise ValueError("distance_metrix is None, cannot determine POI number")

        day_tensors_path = os.path.join(data_dir, 'day_tensors.pt')
        if os.path.exists(day_tensors_path):
            self.day_tensors = torch.load(day_tensors_path)
        else:
            self.day_tensors = torch.zeros((self.num_poi, 7), dtype=torch.float32)
            for poi, data in self.merchant.items():
                for day in data['week']:
                    # Convert 1-based POI index to 0-based index
                    poi_idx = poi - 1
                    if 0 <= poi_idx < self.num_poi:
                        self.day_tensors[poi_idx][day] += 1.0
            self.day_tensors += 0.01 # Smoothing
            self.day_tensors = self.day_tensors / self.day_tensors.sum(dim=1, keepdim=True)
            torch.save(self.day_tensors, day_tensors_path)

        hour_tensors_path = os.path.join(data_dir, 'hour_tensors.pt')
        if os.path.exists(hour_tensors_path):
            self.hour_tensors = torch.load(hour_tensors_path)
        else:
            self.hour_tensors = torch.zeros((self.num_poi, 24), dtype=torch.float32)
            for poi, data in self.merchant.items():
                for hour in data['hour']:
                    # Convert 1-based POI index to 0-based index
                    poi_idx = poi - 1
                    if 0 <= poi_idx < self.num_poi:
                        self.hour_tensors[poi_idx][hour] += 1.0
            self.hour_tensors += 0.01 # Smoothing
            self.hour_tensors = self.hour_tensors / self.hour_tensors.sum(dim=1, keepdim=True)
            torch.save(self.hour_tensors, hour_tensors_path)

        self.beta = beta
        self.candi_merchant_num = candi_merchant_num
        self.hard_negative_ratio = hard_negative_ratio
        
    def run_election(self, poi_embeddings, obs, user_embedding):
        # 支持 batch_size > 1
        batch_size = user_embedding.shape[0] if user_embedding.dim() > 1 else 1
        poi_seq = obs[0].long()
        days = obs[3].long()
        hours = obs[4].long()

        # 1. 用户与 POI 的相似度
        user_embedding_norm = F.normalize(user_embedding, dim=-1)  # [B, D]
        poi_embeddings_norm = F.normalize(poi_embeddings, dim=-1)  # [N, D]
        user_match_score = torch.matmul(user_embedding_norm, poi_embeddings_norm.T)  # [B, N]

        # 2. 用户历史兴趣与 POI 的匹配
        # 对每个 batch 计算历史兴趣
        # Convert 1-based POI indices to 0-based indices
        poi_seq_0based = poi_seq - 1
        # Ensure indices are within bounds
        poi_seq_0based = torch.clamp(poi_seq_0based, 0, poi_embeddings.shape[0] - 1)
        selected_poi_embeddings = poi_embeddings[poi_seq_0based]  # [B, S, D]
        ave_poi_embedding = selected_poi_embeddings.mean(dim=1)  # [B, D]
        poi_match_score = F.cosine_similarity(ave_poi_embedding.unsqueeze(1), poi_embeddings.unsqueeze(0), dim=-1)  # [B, N]

        # 3. 时间分布加权：日
        day_weights = []
        for i in range(batch_size):
            day_tensor = torch.bincount(days[i], minlength=7).float().to(poi_embeddings.device)
            day_tensor = day_tensor / (day_tensor.sum() + 1e-8)
            # 确保 day_tensors 形状正确
            day_tensors_corrected = self.day_tensors.to(day_tensor.device)
            if day_tensors_corrected.shape[0] != poi_embeddings.shape[0]:
                # 如果形状不匹配，调整到正确的POI数量
                day_tensors_corrected = day_tensors_corrected[:poi_embeddings.shape[0]]
            day_weights.append(day_tensors_corrected @ day_tensor)  # [N]
        day_weights = torch.stack(day_weights, dim=0)  # [B, N]

        # 4. 时间分布加权：时
        hour_weights = []
        for i in range(batch_size):
            hour_tensor = torch.bincount(hours[i], minlength=24).float().to(poi_embeddings.device)
            hour_tensor = hour_tensor / (hour_tensor.sum() + 1e-8)
            # 确保 hour_tensors 形状正确
            hour_tensors_corrected = self.hour_tensors.to(hour_tensor.device)
            if hour_tensors_corrected.shape[0] != poi_embeddings.shape[0]:
                # 如果形状不匹配，调整到正确的POI数量
                hour_tensors_corrected = hour_tensors_corrected[:poi_embeddings.shape[0]]
            hour_weights.append(hour_tensors_corrected @ hour_tensor)  # [N]
        hour_weights = torch.stack(hour_weights, dim=0)  # [B, N]

        # 5. 历史序列衰减权重
        poi_weights = []
        for i in range(batch_size):
            T = len(poi_seq[i])
            gamma = 0.8
            decay_weights = gamma ** torch.arange(T - 1, -1, -1, device=poi_seq.device).float()
            decay_weights = decay_weights / (decay_weights.sum() + 1e-8)
            pw = torch.zeros(self.num_poi, dtype=torch.float32, device=poi_seq.device)
            # Convert 1-based POI indices to 0-based indices
            poi_seq_0based = poi_seq[i] - 1
            # Ensure indices are within bounds
            poi_seq_0based = torch.clamp(poi_seq_0based, 0, self.num_poi - 1)
            pw.index_add_(0, poi_seq_0based, decay_weights)
            poi_weights.append(pw)
        poi_weights = torch.stack(poi_weights, dim=0)  # [B, N]

        # 6. 距离加权
        distance_weight = []
        for i in range(batch_size):
            latest_poi = poi_seq[i][-1]
            # Convert 1-based POI index to 0-based index
            latest_poi_0based = latest_poi - 1
            # Ensure index is within bounds
            latest_poi_0based = torch.clamp(latest_poi_0based, 0, self.num_poi - 1)
            distance_vec = self.distance_metrix[latest_poi_0based].to(poi_seq.device)
            distance_weight.append(distance_vec / (distance_vec.max() + 1e-8))
        distance_weight = torch.stack(distance_weight, dim=0)  # [B, N]

        # 1. 基础候选集
        basic_scores = poi_match_score + user_match_score + poi_weights + hour_weights + day_weights + distance_weight
        _, basic_indices = torch.topk(basic_scores, k=int(self.candi_merchant_num * 0.7), dim=1)
        
        # 2. 困难负样本 (Hard Negatives)
        # 选择相似但不是最优的POI作为困难负样本
        similarity_scores = user_match_score
        hard_neg_num = int(self.candi_merchant_num * self.hard_negative_ratio)
        
        # 排除top candidates，选择中等相似度的作为困难负样本
        masked_similarity = similarity_scores.clone()
        for i in range(batch_size):
            masked_similarity[i, basic_indices[i]] = -float('inf')
        
        _, hard_neg_indices = torch.topk(masked_similarity, k=hard_neg_num, dim=1)
        
        # 3. 组合候选集
        combined_indices = torch.cat([basic_indices, hard_neg_indices], dim=1)
        
        return combined_indices