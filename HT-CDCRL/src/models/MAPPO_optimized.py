import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from src.utils.utils import NegotiationEnv, RolloutBuffer, CommitteeElection, create_dataloader_from_buffer, RunningMeanStd
from src.models.modules.Attention import MultiHeadAttentionAd
from src.models.modules.HierarchicalTimeAwareEncoder import HierarchicalTimeAwareEncoder
import pytorch_lightning as pl
import torch
from src.utils.metrics import NormalizedDCG, Hit
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import math
from pathlib import Path
import csv


class OptimizedActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=512, cat_nums=1000, dir='', user_num=0):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),  # 添加dropout
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
        )
        
        self.hier_encoder = HierarchicalTimeAwareEncoder(
            user_num=user_num, 
            cate_size=cat_nums, 
            embed_dim=obs_dim,
            hidden_dim=hidden_dim, 
            poi_num=act_dim
        )

        self.poi_num = act_dim
        
       
        combined_dim = obs_dim + hidden_dim
        self.ranking_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=combined_dim,
                nhead=16,  
                dim_feedforward=combined_dim*4,  
                dropout=0.15,  
                batch_first=True,
                activation='gelu' 
            ), 
            num_layers=4  
        )
        
        # 🔥 增强的排序头 - 残差连接
        self.ranking_head = nn.Sequential(
            nn.Linear(combined_dim, combined_dim//2),
            nn.GELU(),
            nn.LayerNorm(combined_dim//2),  # 🔥 添加LayerNorm
            nn.Dropout(0.1),
            nn.Linear(combined_dim//2, combined_dim//4),
            nn.GELU(),
            nn.Linear(combined_dim//4, 1)
        )
        
        # 🔥 改进的Actor网络
        self.actor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, obs_dim)
        )
        
        # 🔥 增强的Critic网络 - 双头设计
        self.critic_state = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
        )
        
        self.critic_ranking = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim//2),
            nn.Dropout(0.1),
        )
        
        self.critic_head = nn.Linear(hidden_dim, 1)  # hidden_dim//2 + hidden_dim//2
        
        # 🔥 多头注意力机制增强
        self.attention = MultiHeadAttentionAd(hidden_dim, hidden_dim)
        self.attention_norm = nn.LayerNorm(2 * hidden_dim)
        self.dropout = nn.Dropout(p=0.15)
        
        # 🔥 可学习的融合权重
        self.ranking_weight = nn.Parameter(torch.tensor(0.7))
        
        # 🔥 更好的初始化策略
        self._init_weights()

    def _init_weights(self):
        """改进的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 🔥 使用Xavier uniform初始化
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.1)
            elif isinstance(m, nn.TransformerEncoderLayer):
                # 🔥 Transformer层特殊初始化
                for p in m.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)

    def forward(self, merchant_embeddings, merchant_scores, hour, week, cat, poi_index, userid, candidate_poi_ids=None):
        # 获取层次化特征
        feature, _ = self.hier_encoder(hour[:, :-1], week[:, :-1], cat, hour[:, -1], week[:, -1], poi_index, userid)
        
        # 注意力机制
        attn_x, attn_weights = self.attention(feature, merchant_embeddings, merchant_embeddings)
        
        # 🔥 改进的维度处理
        if attn_x.dim() == 4:
            attn_x = attn_x.mean(dim=(1, 2))
        elif attn_x.dim() == 3:
            attn_x = attn_x.mean(dim=1) if attn_x.shape[1] > 1 else attn_x.squeeze(1)
        elif attn_x.dim() > 2:
            attn_x = attn_x.view(attn_x.shape[0], -1)
            
        if feature.dim() > 2:
            feature = feature.view(feature.shape[0], -1)
            
        # 特征融合
        x = torch.cat([attn_x, feature], dim=-1)
        x = self.dropout(x)
        x = self.attention_norm(x)
        
        # 🔥 增强的排序逻辑
        if candidate_poi_ids is not None:
            B, K = candidate_poi_ids.shape
            
            # 获取候选POI嵌入
            flat_poi_ids = candidate_poi_ids.reshape(-1)
            flat_embeddings = self.hier_encoder.poi_embedding(flat_poi_ids)
            flat_embeddings = F.normalize(flat_embeddings, dim=-1)
            candidate_embeddings = flat_embeddings.reshape(B, K, -1)
            
            # 用户表示扩展
            user_expanded = x.unsqueeze(1).repeat(1, K, 1)
            poi_features = candidate_embeddings
            
            # 🔥 特征组合 - 多种交互方式
            combined_features = torch.cat([
                user_expanded[:, :, :poi_features.shape[-1]], 
                poi_features,
            ], dim=-1)
            
            # 🔥 Transformer排序编码 - 加入位置编码
            pos_encoding = self._get_positional_encoding(B, K, combined_features.shape[-1], combined_features.device)
            ranking_input = combined_features + pos_encoding
            ranking_features = self.ranking_encoder(ranking_input)
            
            # 🔥 排序分数计算 - 残差连接
            ranking_scores = self.ranking_head(ranking_features + combined_features).squeeze(-1)
            
            # 🔥 原始logits计算
            original_logits = self.actor(x)
            original_logits = torch.clamp(original_logits, -15, 15)
            original_scores = torch.bmm(original_logits.unsqueeze(1), candidate_embeddings.transpose(1, 2)).squeeze(1)
            
            # 🔥 可学习的权重融合
            alpha = torch.sigmoid(self.ranking_weight)
            logits = alpha * ranking_scores + (1 - alpha) * original_scores
            
            # 🔥 增强的价值函数 - 考虑多种因素
            state_value = self.critic_state(x)
            ranking_quality = ranking_features.mean(dim=1)  # [B, combined_dim]
            ranking_value = self.critic_ranking(ranking_quality[:, :self.hier_encoder.embed_dim])
            
            combined_value_features = torch.cat([state_value, ranking_value], dim=-1)
            value = self.critic_head(combined_value_features).squeeze(-1)
            
        else:
            # 全候选集情况
            all_poi_ids = torch.arange(self.poi_num).to(feature.device)
            all_embeddings = self.hier_encoder.poi_embedding(all_poi_ids)
            all_embeddings = F.normalize(all_embeddings, dim=-1)
            
            original_logits = self.actor(x)
            original_logits = torch.clamp(original_logits, -15, 15)
            logits = torch.matmul(original_logits, all_embeddings.T)
            
            state_value = self.critic_state(x)
            ranking_value = torch.zeros_like(state_value)  # 占位符
            combined_value_features = torch.cat([state_value, ranking_value], dim=-1)
            value = self.critic_head(combined_value_features).squeeze(-1)

        return logits, value
    
    def _get_positional_encoding(self, batch_size, seq_len, d_model, device):
        """🔥 位置编码 - 帮助模型理解候选项顺序"""
        pe = torch.zeros(seq_len, d_model, device=device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0).repeat(batch_size, 1, 1)


class OptimizedMerchantActorCritic(nn.Module):
    """🔥 优化版本的MerchantActorCritic"""
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super().__init__()
        
        # 🔥 增强的编码器
        self.encoder = nn.Sequential(
            nn.Linear(2 * obs_dim, hidden_dim * 2),  # 增加容量
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
        )
        
        # 🔥 改进的Actor和Critic
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim//2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, act_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim//2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, 1)
        )
        
        # 🔥 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, user_embedding, merchant_embeddings):
        # 🔥 改进的维度处理逻辑
        original_user_shape = user_embedding.shape
        original_merchant_shape = merchant_embeddings.shape
        
        # 扁平化处理
        if user_embedding.dim() > 2:
            user_embedding_flat = user_embedding.view(-1, user_embedding.shape[-1])
        else:
            user_embedding_flat = user_embedding
            
        if merchant_embeddings.dim() > 2:
            merchant_embeddings_flat = merchant_embeddings.view(-1, merchant_embeddings.shape[-1])
        else:
            merchant_embeddings_flat = merchant_embeddings
            
        # 确保维度匹配
        if user_embedding_flat.shape[0] != merchant_embeddings_flat.shape[0]:
            if merchant_embeddings_flat.shape[0] % user_embedding_flat.shape[0] == 0:
                repeat_factor = merchant_embeddings_flat.shape[0] // user_embedding_flat.shape[0]
                user_embedding_flat = user_embedding_flat.repeat_interleave(repeat_factor, dim=0)
            elif user_embedding_flat.shape[0] % merchant_embeddings_flat.shape[0] == 0:
                repeat_factor = user_embedding_flat.shape[0] // merchant_embeddings_flat.shape[0]
                merchant_embeddings_flat = merchant_embeddings_flat.repeat_interleave(repeat_factor, dim=0)
            elif user_embedding_flat.shape[0] == 1:
                user_embedding_flat = user_embedding_flat.repeat(merchant_embeddings_flat.shape[0], 1)
            elif merchant_embeddings_flat.shape[0] == 1:
                merchant_embeddings_flat = merchant_embeddings_flat.repeat(user_embedding_flat.shape[0], 1)
            else:
                raise ValueError(f"Cannot match tensor shapes: user_embedding {user_embedding_flat.shape} vs merchant_embeddings {merchant_embeddings_flat.shape}")
        
        # L2归一化
        merchant_embeddings_flat = F.normalize(merchant_embeddings_flat, dim=-1)
        
        # 特征编码
        x = self.encoder(torch.cat([user_embedding_flat, merchant_embeddings_flat], dim=-1))
        
        # Actor和Critic输出
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        
        # 🔥 改进的输出重塑逻辑
        actual_batch_size = user_embedding_flat.shape[0]
        
        if user_embedding.dim() == 4:
            B, S, K = original_user_shape[:3]
            expected_size = B * S * K
            if actual_batch_size == expected_size:
                logits = logits.view(B, S, K, -1)
                value = value.view(B, S, K)
        elif user_embedding.dim() == 3:
            B, K = original_user_shape[:2]
            expected_size = B * K
            if actual_batch_size == expected_size:
                logits = logits.view(B, K, -1)
                value = value.view(B, K)
        elif (user_embedding.dim() == 2 and 
              original_user_shape[0] < original_merchant_shape[0] and 
              original_merchant_shape[0] % original_user_shape[0] == 0):
            B = original_user_shape[0]
            K = original_merchant_shape[0] // B
            if actual_batch_size == B * K:
                logits = logits.view(B, K, -1)
                value = value.view(B, K)
        
        return logits, value


class OptimizedUserMAPPOAgent(torch.nn.Module):
    """🔥 优化版本的UserMAPPOAgent"""
    def __init__(self, long_term, obs_dim, act_dim, poi_num, user_num, lr=3e-4, gamma=0.99, clip_ratio=0.2, batch_size=8, cat_num=1000, dir='', seed=42, *args, **kwargs):
        super().__init__(*args, **kwargs)
        torch.manual_seed(seed)
        
        self.policy_net = OptimizedActorCritic(obs_dim, act_dim, hidden_dim=obs_dim, cat_nums=cat_num, dir=dir, user_num=user_num)
        self.optimizer = None
        self.scheduler = None  # 🔥 添加学习率调度器
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.poi_num = poi_num
        self.batch_size = batch_size
        self.seed = seed
        
        # 由 Lightning 管理混合精度，不自建 GradScaler
        self.scaler = None

    def select_action(self, merchant_embeddings, merchant_act_embeddings, x, hour, week, cat, userid, candidate_poi_ids=None):
        return self.policy_net(
            merchant_embeddings, merchant_act_embeddings,
            hour.to(torch.int64), week.to(torch.int64), cat.to(torch.int64), x, userid, candidate_poi_ids
        )

    def update(self, buffer, poi_embedding, committee, merchant, current_epoch, pl_module):
        dataloader = create_dataloader_from_buffer(buffer, agent_type='user', batch_size=self.batch_size)
        
        losses, policy_losses, value_losses, entropies, ranking_losses = [], [], [], [], []
        
        for mul_obs, act, other_act, reward, mul_next_obs, user_embedding, merchats_embeddings, selected_action in tqdm(dataloader, desc="Updating User Agent", leave=False):
            # 🔥 数据预处理增强
            reward = torch.nan_to_num(reward, nan=0.0, posinf=10.0, neginf=-10.0)
            reward = torch.clamp(reward, min=-10, max=10)
            user_embedding = torch.nan_to_num(user_embedding, nan=0.0)
            merchats_embeddings = torch.nan_to_num(merchats_embeddings, nan=0.0)

            x = torch.stack([obs[0] for obs in mul_obs]).squeeze(1).squeeze(1)
            hour = torch.stack([obs[4] for obs in mul_obs]).squeeze(1).squeeze(1)
            week = torch.stack([obs[3] for obs in mul_obs]).squeeze(1).squeeze(1)
            cat = torch.stack([obs[1] for obs in mul_obs]).squeeze(1).squeeze(1)
            userid = torch.stack([obs[5] for obs in mul_obs]).squeeze(1)

            # 由 Lightning 统一管理 autocast
            logits, values = self.select_action(merchats_embeddings.squeeze(1).detach(), act, x, hour, week, cat[:, :-1], userid)
            
            # 🔥 改进的概率计算
            logits = torch.clamp(logits, -25, 25)  # 扩大范围
            probs_new = F.softmax(logits, dim=1)
            probs_new = torch.clamp(probs_new, 1e-8, 1.0)
            
            dist_new = Categorical(probs=probs_new)
            
            # 动作处理
            batch_size, num_candidates = other_act.shape
            selected_action_clamped = selected_action.to(torch.int64).clamp(0, num_candidates - 1)
            
            log_prob_new = dist_new.log_prob(selected_action_clamped)
            log_prob_old = torch.log(other_act.gather(1, selected_action_clamped.unsqueeze(1)).squeeze(1) + 1e-8).detach()

            # 🔥 计算下一状态值
            selected_merchants_list = []
            for batch_idx, obs in enumerate(mul_next_obs):
                obs_user_embedding = obs[-1]
                if obs_user_embedding.dim() > 1 and obs_user_embedding.shape[0] > 1:
                    obs_user_embedding = obs_user_embedding[batch_idx:batch_idx+1]
                elif obs_user_embedding.dim() == 1:
                    obs_user_embedding = obs_user_embedding.unsqueeze(0)
                    
                selected_merchant_index = committee.run_election(poi_embedding, obs, obs_user_embedding)
                selected_merchants = poi_embedding[selected_merchant_index.to(logits.device)]
                selected_merchants_list.append(selected_merchants)

            hour_next = torch.stack([obs[4] for obs in mul_next_obs]).squeeze(1).squeeze(1)
            week_next = torch.stack([obs[3] for obs in mul_next_obs]).squeeze(1).squeeze(1)
            cat_next = torch.stack([obs[1] for obs in mul_next_obs]).squeeze(1).squeeze(1)
            x_next = torch.stack([obs[0] for obs in mul_next_obs]).squeeze(1).squeeze(1)
            selected_merchants = torch.stack(selected_merchants_list, dim=0)
            
            user_embedding_squeezed = user_embedding.squeeze(1)
            user_embedding_expanded = user_embedding_squeezed.unsqueeze(1).repeat(1, selected_merchants.shape[1], 1)

            actions_merchant, _ = merchant.select_action_batch(user_embedding_expanded, selected_merchants)
            
            _, next_values = self.select_action(selected_merchants, actions_merchant, x_next, hour_next, week_next, cat_next[:, 1:], userid)

            # 🔥 改进的优势函数计算 - GAE
            with torch.no_grad():
                advantage = reward + (self.gamma * next_values - values)
                # 可以在这里添加GAE逻辑

            # 🔥 损失函数计算
            value_loss = F.mse_loss(values, (reward + self.gamma * next_values).detach())

            log_ratio = torch.clamp(log_prob_new - log_prob_old, -15, 15)
            ratio = torch.exp(log_ratio)
            surrogate = ratio * advantage.detach()
            clipped = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage.detach()
            policy_loss = -torch.min(surrogate, clipped).mean()
            
            # 🔥 增强的排序损失
            positive_probs = probs_new.gather(1, selected_action_clamped.unsqueeze(1))
            mask = torch.ones_like(probs_new, dtype=torch.bool)
            mask.scatter_(1, selected_action_clamped.unsqueeze(1), False)
            negative_probs = probs_new[mask].view(probs_new.size(0), -1)
            
            # 🔥 改进的BPR Loss + Margin Loss
            margin_loss = torch.clamp(0.5 - (positive_probs - negative_probs.max(dim=1, keepdim=True)[0]), min=0).mean()
            ranking_loss = -F.logsigmoid(positive_probs - negative_probs.mean(dim=1, keepdim=True)).mean()
            combined_ranking_loss = ranking_loss + 0.1 * margin_loss
            
            entropy = dist_new.entropy().mean()
            
            # 🔥 总损失 - 动态权重
            entropy_weight = max(0.005, 0.02 * (1 - current_epoch / 200))  # 递减的熵权重
            ranking_weight = min(0.3, 0.1 + 0.2 * (current_epoch / 100))   # 递增的排序权重
            
            loss = policy_loss + 0.5 * value_loss - entropy_weight * entropy + ranking_weight * combined_ranking_loss

            losses.append(loss.item())
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.item())
            ranking_losses.append(combined_ranking_loss.item())

            # 🔥 优化器步骤
            self.optimizer.zero_grad()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
        
        # 日志记录
        if losses:
            avg_loss = sum(losses) / len(losses)
            pl_module.log("train/user_policy_loss_epoch", sum(policy_losses) / len(policy_losses), on_step=False, on_epoch=True)
            pl_module.log("train/user_value_loss_epoch", sum(value_losses) / len(value_losses), on_step=False, on_epoch=True)
            pl_module.log("train/user_entropy_epoch", sum(entropies) / len(entropies), on_step=False, on_epoch=True)
            pl_module.log("train/user_ranking_loss_epoch", sum(ranking_losses) / len(ranking_losses), on_step=False, on_epoch=True)
            return avg_loss
        return None


class OptimizedMerchantMAPPOAgent(torch.nn.Module):
    """🔥 优化版本的MerchantMAPPOAgent"""
    def __init__(self, obs_dim, act_dim, poi_num, lr=3e-4, gamma=0.99, clip_ratio=0.2, batch_size=8, seed=42, *args, **kwargs):
        super().__init__(*args, **kwargs)
        torch.manual_seed(seed)
        
        self.policy_net = OptimizedMerchantActorCritic(obs_dim, act_dim, hidden_dim=obs_dim)
        self.optimizer = None
        self.scheduler = None  # 🔥 添加学习率调度器
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.poi_num = poi_num if poi_num is not None else 1
        self.batch_size = batch_size
        self.seed = seed
        
        # 由 Lightning 管理混合精度，不自建 GradScaler
        self.scaler = None

    def select_action_batch(self, user_embedding, merchant_embeddings):
        logits, values = self.policy_net(user_embedding, merchant_embeddings)
        return torch.sigmoid(logits), values

    def update(self, buffer, committee, poi_table, pl_module):
        dataloader = create_dataloader_from_buffer(buffer, agent_type='merchant', batch_size=self.batch_size)
        
        losses, policy_losses, value_losses = [], [], []
        epsilon = 1e-8

        for mul_obs, act, other_act, reward, mul_next_obs, user_embedding, merchats_embeddings, selected_action in tqdm(dataloader, desc="Updating Merchant Agent", leave=False):
            # 🔥 数据预处理
            reward = torch.nan_to_num(reward, nan=0.0, posinf=10.0, neginf=-10.0).clamp(min=-10.0, max=10.0)
            user_embedding = torch.nan_to_num(user_embedding, nan=0.0)
            merchats_embeddings = torch.nan_to_num(merchats_embeddings, nan=0.0)

            user_embedding_squeezed = user_embedding.squeeze(1)
            user_embedding_expanded = user_embedding_squeezed.unsqueeze(1).repeat(1, merchats_embeddings.shape[1], 1)

            # 由 Lightning 统一管理 autocast
            logits, values = self.select_action_batch(user_embedding_expanded, merchats_embeddings.detach())
            
            logits = torch.clamp(logits, min=epsilon, max=1 - epsilon)

            dist = torch.distributions.Bernoulli(probs=logits)
            log_prob_new = dist.log_prob(selected_action.to(logits.device).float())
            act = torch.clamp(act, min=epsilon, max=1 - epsilon)
            log_prob_old = torch.log(act + epsilon).detach()

            # 🔥 奖励维度匹配
            if reward.shape != values.shape:
                if reward.dim() == 1 and values.dim() == 2:
                    reward = reward.unsqueeze(1).expand_as(values)

            # 🔥 计算损失
            with torch.no_grad():
                next_value = values.detach() * 0.95

            advantage = reward + self.gamma * next_value - values

            value_loss = F.mse_loss(values, (reward + self.gamma * next_value).detach())
            
            log_ratio = torch.clamp(log_prob_new - log_prob_old, -15, 15)
            ratio = torch.exp(log_ratio)
            surrogate = ratio * advantage.detach().unsqueeze(-1)
            clipped = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage.detach().unsqueeze(-1)
            policy_loss = -torch.min(surrogate, clipped).mean()

            loss = policy_loss + 0.5 * value_loss

            losses.append(loss.item())
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

            # 🔥 优化器步骤
            self.optimizer.zero_grad()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
        
        # 日志记录
        if losses:
            avg_loss = sum(losses) / len(losses)
            pl_module.log("train/merchant_policy_loss_epoch", sum(policy_losses) / len(policy_losses), on_step=False, on_epoch=True)
            pl_module.log("train/merchant_value_loss_epoch", sum(value_losses) / len(value_losses), on_step=False, on_epoch=True)
            return avg_loss
        return None


class OptimizedMAPPO(pl.LightningModule):
    """🔥 优化版本的MAPPO主类"""
    def __init__(self, data_dir, obs_dim, topk, batch_size=8, committee_weight=0.7, lr=1e-3, weight_decay=1e-3, update_steps=1, candi_merchant_num=10, cat_num=1000, reward_alpha=1.0, reward_beta=0.1, reward_gamma=0.1, clip_ratio=0.2, seed=42, ranking_alpha: float = 0.7, logging_manager=None):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        
        # 🔥 初始化日志管理器
        self.logging_manager = logging_manager
        
        # 🔥 加载数据
        distance_matrix_path = os.path.join(data_dir, 'distance_matrix.pt')
        if not os.path.exists(distance_matrix_path):
             raise FileNotFoundError(f"distance_matrix.pt not found in {data_dir}")
        distance_metrix = torch.load(distance_matrix_path)
        self.poi_num = distance_metrix.shape[0]

        long_term_path = os.path.join(data_dir, 'long_term.pkl')
        self.long_term = torch.load(long_term_path) if os.path.exists(long_term_path) else {}
        self.user_num = len(self.long_term)

        poi_long_term_path = os.path.join(data_dir, 'poi_long_term_after_process.pkl')
        self.poi_long_term = torch.load(poi_long_term_path) if os.path.exists(poi_long_term_path) else {}

        # 🔥 实例化优化的组件
        self.user_mappo = OptimizedUserMAPPOAgent(
            user_num=self.user_num, 
            dir=data_dir, 
            cat_num=cat_num, 
            batch_size=batch_size, 
            poi_num=self.poi_num, 
            long_term=self.long_term, 
            obs_dim=obs_dim, 
            act_dim=self.poi_num, 
            lr=self.lr,
            clip_ratio=clip_ratio,
            seed=seed
        )
        # 将配置中的 ranking_alpha 应用于可学习融合权重
        try:
            import math as _math
            ranking_alpha_clamped = max(1e-6, min(1.0 - 1e-6, float(ranking_alpha)))
            init_weight = _math.log(ranking_alpha_clamped / (1.0 - ranking_alpha_clamped))
            with torch.no_grad():
                if hasattr(self.user_mappo.policy_net, 'ranking_weight'):
                    self.user_mappo.policy_net.ranking_weight.copy_(torch.tensor(init_weight, dtype=self.user_mappo.policy_net.ranking_weight.dtype, device=self.user_mappo.policy_net.ranking_weight.device))
        except Exception:
            pass
        
        self.merchant_mappo = OptimizedMerchantMAPPOAgent(
            batch_size=batch_size, 
            poi_num=self.poi_num, 
            obs_dim=obs_dim, 
            act_dim=1, 
            lr=self.lr,
            clip_ratio=clip_ratio,
            seed=seed
        )
        
        self.committee = CommitteeElection(data_dir, distance_metrix=distance_metrix, poi_long_term=self.poi_long_term, beta=committee_weight, candi_merchant_num=candi_merchant_num, seed=seed)
        self.env = NegotiationEnv(
            self.user_mappo.policy_net.hier_encoder.poi_embedding,
            distance_matrix=distance_metrix,
            alpha=self.hparams.reward_alpha,
            beta=self.hparams.reward_beta,
            gamma=self.hparams.reward_gamma
        )
        self.buffer = RolloutBuffer(data_dir)

        # 评价指标
        self.topk = topk
        self.NDCG = nn.ModuleDict({str(k): NormalizedDCG(k) for k in self.topk})
        self.HIT = nn.ModuleDict({str(k): Hit(k) for k in self.topk})
        self.NDCG_val = nn.ModuleDict({str(k): NormalizedDCG(k) for k in self.topk})
        self.HIT_val = nn.ModuleDict({str(k): Hit(k) for k in self.topk})
        
        self.episode_rewards_user = []
        self.episode_rewards_merchant = []
        self.batch_counter = 0
        self.update_steps = update_steps
        self.base_size = 1200  # 🔥 增加基础大小
        self.increment = 600   # 🔥 增加增量
        self.reward_normalizer = RunningMeanStd(shape=())

        # ========== 日志与CSV路径（区分数据集/模型/实验） ==========
        _dataset_name = os.path.basename(self.hparams.data_dir) if hasattr(self, 'hparams') else 'dataset'
        _model_name = self.__class__.__name__
        # 从 logger 中取实验名，取不到则用默认
        _experiment = None
        try:
            _experiment = getattr(self.logger, 'name', None)
        except Exception:
            _experiment = None
        if not _experiment:
            _experiment = 'default_exp'
        # run_id：优先用环境中的 WANDB_RUN_ID/SWANLAB_RUN_ID，否则时间戳
        import datetime
        _run_id = os.environ.get('WANDB_RUN_ID') or os.environ.get('SWANLAB_RUN_ID') or datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        self._meta_dataset = _dataset_name
        self._meta_model = _model_name
        self._meta_experiment = _experiment
        self._meta_run_id = _run_id

        self.save_root = Path("logs") / _dataset_name / _model_name / _experiment / _run_id
        self.preds_dir = self.save_root / "pred_distributions"
        self.emb_dir = self.save_root / "user_embeddings"
        self.csv_root = self.save_root / "csv"
        for d in [self.preds_dir, self.emb_dir, self.csv_root]:
            d.mkdir(parents=True, exist_ok=True)
        self._pid = os.getpid()
        # CSV files
        self.csv_preds_val = self.csv_root / "preds_val.csv"
        self.csv_preds_test = self.csv_root / "preds_test.csv"
        self.csv_embeddings = self.csv_root / "user_embeddings.csv"
        self.csv_rewards = self.csv_root / "rewards.csv"
        # 临时缓存本epoch用户表征
        self._epoch_user_embeddings = []

        # ====== 课程学习与融合权重调度 ======
        self.curriculum_stages = [
            {"epochs": 15, "candi_num": max(10, int(candi_merchant_num * 0.8))},
            {"epochs": 40, "candi_num": max(15, int(candi_merchant_num))},
            {"epochs": 999, "candi_num": max(20, int(candi_merchant_num + 5))},
        ]
        self.initial_candi_num = candi_merchant_num
        # ranking_alpha目标随epoch线性从起始值缓慢提升到0.8
        self.ranking_alpha_start = 0.55
        self.ranking_alpha_target = 0.80
        self.ranking_alpha_warm_epochs = 50

    def _append_csv(self, csv_path: Path, fieldnames, rows):
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = csv_path.exists()
        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            if isinstance(rows, dict):
                writer.writerow(rows)
            else:
                writer.writerows(rows)

    def configure_optimizers(self):
        """🔥 优化的优化器和调度器配置"""
        # 创建优化器
        user_optimizer = Adam(
            self.user_mappo.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),  # 🔥 调整beta参数
            eps=1e-8
        )
        
        merchant_optimizer = Adam(
            self.merchant_mappo.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # 🔥 创建学习率调度器
        user_scheduler = OneCycleLR(
            user_optimizer,
            max_lr=self.hparams.lr * 2,  # 最大学习率是初始的2倍
            total_steps=self.trainer.estimated_stepping_batches if hasattr(self.trainer, 'estimated_stepping_batches') else 10000,
            pct_start=0.1,  # 10%时间用于warmup
            anneal_strategy='cos',
            div_factor=25,  # 初始lr = max_lr / div_factor
            final_div_factor=1000,  # 最终lr = max_lr / final_div_factor
        )
        
        merchant_scheduler = OneCycleLR(
            merchant_optimizer,
            max_lr=self.hparams.lr * 2,
            total_steps=self.trainer.estimated_stepping_batches if hasattr(self.trainer, 'estimated_stepping_batches') else 10000,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=25,
            final_div_factor=1000,
        )

        # 分配给MAPPO实例
        self.user_mappo.optimizer = user_optimizer
        self.user_mappo.scheduler = user_scheduler
        self.merchant_mappo.optimizer = merchant_optimizer
        self.merchant_mappo.scheduler = merchant_scheduler

        return user_optimizer

    def on_train_start(self) -> None:
        self.automatic_optimization = False
        # 首次应用课程难度与融合权重
        self._adjust_curriculum_difficulty()
        self._schedule_ranking_alpha()

    def on_train_epoch_start(self) -> None:
        # 每个epoch开始时更新课程难度与融合权重
        self._adjust_curriculum_difficulty()
        self._schedule_ranking_alpha()

    def training_step(self, batch, batch_idx):
        # 🔥 动态训练长度
        max_len = getattr(self.trainer.train_dataloader, 'len', float('inf'))
        dynamic_max = min(max_len, self.base_size + self.current_epoch * self.increment)
        
        if self.batch_counter > dynamic_max:
            return None
        self.batch_counter += 1

        # 🔥 现有训练逻辑保持不变，但所有组件都已优化
        x, cat, y, hour, week, userid, hour_pre, week_pre, _, _, user_hash, poi_hash = batch.values()
        batch_size = x.shape[0]

        obs = [x, cat, None, torch.cat([week, week_pre[:, -1].unsqueeze(-1)], dim=-1), torch.cat([hour, hour_pre[:, -1].unsqueeze(-1)], dim=-1), userid, user_hash, poi_hash]

        all_poi_ids = torch.arange(self.poi_num, device=self.device)
        all_poi_embeddings = self.user_mappo.policy_net.hier_encoder.poi_embedding(all_poi_ids)

        user_embedding, _ = self.user_mappo.policy_net.hier_encoder(hour, week, cat[:, :-1], hour_pre[:, -1], week_pre[:, -1], x, userid)
        obs.append(user_embedding.detach())
        
        # 收集用于epoch级CSV/pt保存
        try:
            self._epoch_user_embeddings.append(user_embedding.detach())
        except Exception:
            pass

        selected_merchant_index = self.committee.run_election(all_poi_embeddings.detach(), obs, user_embedding.detach())

        action_user_probs_list = []
        reward_users_list = []
        reward_merchants_list = []
        merchant_embeddings_list = []

        for i in range(batch_size):
            single_merchant_index = selected_merchant_index[i]
            merchant_embeddings = self.user_mappo.policy_net.hier_encoder.poi_embedding(single_merchant_index)
            merchant_embeddings_list.append(merchant_embeddings)

            user_embedding_single = user_embedding[i:i+1].repeat(merchant_embeddings.size(0), 1)
            actions_merchant, _ = self.merchant_mappo.select_action_batch(user_embedding_single, merchant_embeddings)

            with torch.no_grad():
                action_user_logits, _ = self.user_mappo.select_action(
                    merchant_embeddings, actions_merchant,
                    x[i:i+1], obs[3][i:i+1], obs[4][i:i+1],
                    cat[i:i+1, :-1], userid[i:i+1],
                    candidate_poi_ids=single_merchant_index.unsqueeze(0)
                )

            action_user_logits = action_user_logits.squeeze(0)
            action_user_probs = F.softmax(action_user_logits, dim=-1)

            obs_single = []
            for j in range(len(obs)):
                if obs[j] is None:
                    obs_single.append(None)
                elif isinstance(obs[j], torch.Tensor):
                    obs_single.append(obs[j][i:i+1])
                else:
                    obs_single.append([obs[j][i]])

            _, reward_user, reward_merchant = self.env.step(
                action_user_probs, single_merchant_index, actions_merchant,
                obs_single, None
            )

            action_user_probs_list.append(action_user_probs)
            reward_users_list.append(reward_user)
            reward_merchants_list.append(reward_merchant)

        reward_users = torch.stack(reward_users_list, dim=0)
        reward_merchants = torch.stack(reward_merchants_list, dim=0)
        
        # 🔥 保存batch激励到日志管理器
        if self.logging_manager is not None:
            try:
                self.logging_manager.save_batch_rewards(
                    user_rewards=reward_users,
                    merchant_rewards=reward_merchants,
                    epoch=self.current_epoch,
                    batch_idx=batch_idx
                )
            except Exception as e:
                print(f"Warning: Failed to save batch rewards: {e}")

        # 🔥 其余训练逻辑保持类似，但添加了更多正则化和监控
        consistency_loss = 0.0
        diversity_loss = 0.0

        for i in range(batch_size):
            action_user_probs = action_user_probs_list[i]
            entropy = -torch.sum(action_user_probs * torch.log(action_user_probs + 1e-8))
            max_entropy = torch.log(torch.tensor(len(action_user_probs), dtype=torch.float, device=self.device))
            entropy_loss = -(entropy / max_entropy)

            user_hist_pois = obs[0][i].long().flatten()
            if len(user_hist_pois) > 0:
                hist_embeddings = self.user_mappo.policy_net.hier_encoder.poi_embedding(user_hist_pois)
                candidate_embeddings = self.user_mappo.policy_net.hier_encoder.poi_embedding(selected_merchant_index[i])
                similarities = F.cosine_similarity(
                    candidate_embeddings.unsqueeze(1),
                    hist_embeddings.unsqueeze(0),
                    dim=-1
                ).mean(dim=1)
                consistency_reward = torch.sum(action_user_probs * similarities)
                consistency_loss += -consistency_reward

            diversity_loss += entropy_loss

        consistency_loss = consistency_loss / batch_size
        diversity_loss = diversity_loss / batch_size
        reward_users = torch.nan_to_num(reward_users, nan=0.0, posinf=1.0, neginf=-1.0)
        next_obs = [y, cat, None, obs[3], obs[4], userid, user_hash, poi_hash, user_embedding.detach()]
        random_candidate_index = torch.randint(0, len(selected_merchant_index[0]), (1,), device=self.device)[0]

        obs_first = []
        for j in range(len(obs)):
            if obs[j] is None:
                obs_first.append(None)
            elif isinstance(obs[j], torch.Tensor):
                obs_first.append(obs[j][0:1])
            else:
                obs_first.append([obs[j][0]])

        next_obs_first = []
        for j in range(len(next_obs)):
            if next_obs[j] is None:
                next_obs_first.append(None)
            elif isinstance(next_obs[j], torch.Tensor):
                next_obs_first.append(next_obs[j][0:1])
            else:
                next_obs_first.append([next_obs[j][0]])
        
        first_merchant_embeddings = merchant_embeddings_list[0]
        self.buffer.store(batch_idx, obs_first, actions_merchant.detach(), action_user_probs_list[0].detach(), reward_users[0].detach(), next_obs_first, user_embedding[0:1].detach(), first_merchant_embeddings.detach(), random_candidate_index, agent_type='user')
        merchant_action_binary = (actions_merchant.detach() > 0.5).float()
        self.buffer.store(batch_idx, obs_first, actions_merchant.detach(), action_user_probs_list[0].detach(), reward_merchants[0].detach(), next_obs_first, user_embedding[0:1].detach(), first_merchant_embeddings.detach(), merchant_action_binary, agent_type='merchant')

        # 🔥 增强的日志记录
        self.log("train/user_reward", reward_users.mean(), prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/merchant_reward", reward_merchants.mean(), prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/consistency_loss", consistency_loss, prog_bar=True)
        self.log("train/diversity_loss", diversity_loss, prog_bar=True)

        
        # 🔥 添加更多监控指标
        if hasattr(self.user_mappo.policy_net, 'ranking_weight'):
            self.log("train/ranking_alpha", torch.sigmoid(self.user_mappo.policy_net.ranking_weight), prog_bar=False)

        if batch_idx % 50 == 0:  # 更频繁的调试输出
            print(f"\n=== Optimized Training Step Debug Info (Batch {batch_idx}) ===")
            print(f"Batch size: {batch_size}")
            print(f"Candidate set size: {selected_merchant_index.shape[1]}")
            print(f"User reward mean: {reward_users.mean().item():.4f}")
            print(f"Merchant reward mean: {reward_merchants.mean().item():.4f}")
            print(f"Action probs sample: {action_user_probs_list[0][:5].tolist()}")
            print(f"Selected candidates sample: {selected_merchant_index[0][:5].tolist()}")
            print("=== Optimized Training ===\n")

    def on_train_epoch_end(self) -> None:
        # 🔥 优化的epoch结束处理
        # 使用LoggingManager保存用户表征
        if len(self._epoch_user_embeddings) > 0 and self.logging_manager is not None:
            try:
                epoch_emb = torch.cat(self._epoch_user_embeddings, dim=0).detach()
                # 生成用户ID（这里使用索引，实际应用中可能需要真实的用户ID）
                user_ids = torch.arange(len(epoch_emb), device=self.device)
                
                self.logging_manager.save_user_embeddings(
                    user_embeddings=epoch_emb,
                    user_ids=user_ids,
                    epoch=int(self.current_epoch)
                )
            except Exception as e:
                print(f"Warning: Failed to save user embeddings: {e}")
            finally:
                self._epoch_user_embeddings.clear()

        if not self.buffer or len(self.buffer) == 0:
            print("Buffer is empty. Skipping epoch end update.")
            return

        self.buffer.merge_data()
        all_poi_ids = torch.arange(self.poi_num, device=self.device)
        
        # 🔥 用户agent更新
        user_total_loss = 0
        for i in range(self.update_steps):
            print(f"Epoch {self.current_epoch}, Optimized User Update Step {i+1}/{self.update_steps}")
            poi_embedding = self.user_mappo.policy_net.hier_encoder.poi_embedding(all_poi_ids)
            loss = self.user_mappo.update(self.buffer, committee=self.committee, merchant=self.merchant_mappo, poi_embedding=poi_embedding, current_epoch=self.current_epoch, pl_module=self)
            if loss: user_total_loss += loss
            
        if self.update_steps > 0 and user_total_loss is not None:
            self.log("train/user_loss_epoch", user_total_loss / self.update_steps, on_step=False, on_epoch=True, prog_bar=True)
            
        # 🔥 商家agent更新
        merchant_total_loss = 0
        for i in range(self.update_steps):
            print(f"Epoch {self.current_epoch}, Optimized Merchant Update Step {i+1}/{self.update_steps}")
            loss = self.merchant_mappo.update(self.buffer, committee=self.committee, poi_table=self.user_mappo.policy_net.hier_encoder.poi_embedding, pl_module=self)
            if loss: merchant_total_loss += loss
            
        if self.update_steps > 0 and merchant_total_loss is not None:
            self.log("train/merchant_loss_epoch", merchant_total_loss / self.update_steps, on_step=False, on_epoch=True, prog_bar=True)
            
        self.buffer.clear()
        self.batch_counter = 0

    def _shared_eval_step(self, batch, batch_idx, prefix):
        # 🔥 保持原有评估逻辑，但使用优化的组件
        x, cat, y, hour, week, userid, hour_pre, week_pre, _, _, _, _ = batch.values()
        
        user_embedding, _ = self.user_mappo.policy_net.hier_encoder(hour, week, cat[:, :-1], hour_pre[:, -1], week_pre[:, -1], x, userid)
        
        batch_size = y.shape[0]
        max_topk = max(self.topk)
        
        original_candi_num = self.committee.candi_merchant_num
        self.committee.candi_merchant_num = max_topk - 1
        
        obs = [x, cat, None, torch.cat([week, week_pre[:, -1].unsqueeze(-1)], dim=-1), 
               torch.cat([hour, hour_pre[:, -1].unsqueeze(-1)], dim=-1), userid, None, None]
        obs.append(user_embedding.detach())
        
        all_poi_ids = torch.arange(self.poi_num, device=self.device)
        all_poi_embeddings = self.user_mappo.policy_net.hier_encoder.poi_embedding(all_poi_ids)
        selected_merchant_index = self.committee.run_election(all_poi_embeddings.detach(), obs, user_embedding.detach())
        
        self.committee.candi_merchant_num = original_candi_num
        
        eval_K = max_topk
        final_candidate_ids = torch.zeros(batch_size, eval_K, dtype=torch.long, device=self.device)
        insert_positions = torch.randint(0, eval_K, (batch_size,), device=self.device)
        
        for i in range(batch_size):
            pos = insert_positions[i]
            candidates = selected_merchant_index[i]
            gt_poi = y[i, -1].item()
            
            if gt_poi in candidates:
                remaining_mask = torch.ones(self.poi_num, dtype=torch.bool, device=self.device)
                remaining_mask[candidates] = False
                remaining_mask[gt_poi] = False
                remaining = remaining_mask.nonzero(as_tuple=False).squeeze(-1)
                if len(remaining) > 0:
                    replacement = remaining[torch.randint(0, len(remaining), (1,))]
                    candidates = torch.where(candidates == gt_poi, replacement, candidates)
            
            if pos == 0:
                final_candidate_ids[i] = torch.cat([y[i, -1].unsqueeze(0), candidates[:eval_K-1]])
            elif pos == eval_K - 1:
                final_candidate_ids[i] = torch.cat([candidates[:eval_K-1], y[i, -1].unsqueeze(0)])
            else:
                final_candidate_ids[i] = torch.cat([
                    candidates[:pos], 
                    y[i, -1].unsqueeze(0), 
                    candidates[pos:eval_K-1]
                ])
            
        all_poi_embeddings = self.user_mappo.policy_net.hier_encoder.poi_embedding(final_candidate_ids)
        poi_action_embeddings, _ = self.merchant_mappo.select_action_batch(
            user_embedding.unsqueeze(1).repeat(1, eval_K, 1), all_poi_embeddings
        )
        
        action_user, _ = self.user_mappo.select_action(
            all_poi_embeddings, poi_action_embeddings, x,
            torch.cat([hour, hour_pre[:, -1].unsqueeze(-1)], dim=-1),
            torch.cat([week, week_pre[:, -1].unsqueeze(-1)], dim=-1),
            cat[:, :-1], userid, candidate_poi_ids=final_candidate_ids
        )
        
        preds = torch.sigmoid(action_user)
        Ground = F.one_hot(insert_positions.long(), num_classes=eval_K).float()

        # 🔥 使用LoggingManager保存预测分布
        if self.logging_manager is not None:
            try:
                self.logging_manager.save_prediction_distributions(
                    preds=preds,
                    candidate_ids=final_candidate_ids,
                    gt_positions=insert_positions,
                    user_ids=userid,
                    epoch=int(self.current_epoch),
                    batch_idx=int(batch_idx),
                    split=prefix
                )
            except Exception as e:
                print(f"Warning: Failed to save prediction distributions: {e}")

            B, K = preds.shape
            rows = []
            for i in range(B):
                row = {
                    "split": prefix,
                    "epoch": int(self.current_epoch),
                    "pid": int(self._pid),
                    "batch_idx": int(batch_idx),
                    "sample_idx": int(i),
                    "gt_pos": int(insert_positions[i].item()),
                }
                try:
                    row["user_id"] = int(userid[i].item())
                except Exception:
                    pass
                for k in range(K):
                    row[f"cand_{k}"] = int(final_candidate_ids[i, k].item())
                    row[f"pred_{k}"] = float(preds[i, k].item())
                rows.append(row)
            fieldnames = [
                "split", "epoch", "pid", "batch_idx", "sample_idx", "gt_pos", "user_id",
            ] + [f"cand_{k}" for k in range(K)] + [f"pred_{k}" for k in range(K)]
            dst = self.csv_preds_val if prefix == "val" else self.csv_preds_test
            self._append_csv(dst, fieldnames, rows)
        
        for k_str, metric in (self.NDCG if prefix == "test" else self.NDCG_val).items():
            k_val = int(k_str)
            if k_val <= eval_K:
                metric.update(preds, Ground)
                self.log(f"{prefix}/NDCG@{k_str}", metric, on_step=False, on_epoch=True)
            
        for k_str, metric in (self.HIT if prefix == "test" else self.HIT_val).items():
            k_val = int(k_str)
            if k_val <= eval_K:
                metric.update(preds, Ground)
                self.log(f"{prefix}/Hit@{k_str}", metric, on_step=False, on_epoch=True, prog_bar=True)
        
        if batch_idx % 50 == 0:  # 更频繁的调试输出
            print(f"\n=== Optimized {prefix.capitalize()} Step Debug Info (Batch {batch_idx}) ===")
            print(f"Max topk: {max_topk}, Eval_K: {eval_K}")
            print(f"Candidate set size: {final_candidate_ids.shape[1]}")
            print(f"Ground truth position sample: {insert_positions[:5].tolist()}")
            print(f"Preds shape: {preds.shape}")
            print(f"Ground shape: {Ground.shape}")
            print("=== Optimized Evaluation Mode ===\n")

    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval_step(batch, batch_idx, "test")

    def setup(self, stage=None):
        """🔥 优化的setup方法"""
        if not hasattr(self.user_mappo, 'optimizer') or not hasattr(self.merchant_mappo, 'optimizer'):
            # 重新配置优化器
            self.configure_optimizers()

    def _adjust_curriculum_difficulty(self):
        current_epoch = int(self.current_epoch)
        for stage in self.curriculum_stages:
            if current_epoch < stage["epochs"]:
                new_candi = stage["candi_num"]
                if hasattr(self.committee, 'candi_merchant_num') and self.committee.candi_merchant_num != new_candi:
                    old = self.committee.candi_merchant_num
                    self.committee.candi_merchant_num = new_candi
                    print(f"Epoch {current_epoch}: Adjust candidate POI num {old} -> {new_candi}")
                break

    def _schedule_ranking_alpha(self):
        # 线性从 start -> target，持续 warm_epochs
        e = float(self.current_epoch)
        t = min(1.0, max(0.0, e / max(1.0, float(self.ranking_alpha_warm_epochs))))
        target_alpha = (1.0 - t) * float(self.ranking_alpha_start) + t * float(self.ranking_alpha_target)
        target_alpha = max(1e-6, min(1.0 - 1e-6, target_alpha))
        # 写入可学习权重的logit值
        try:
            import math as _math
            logit = _math.log(target_alpha / (1.0 - target_alpha))
            with torch.no_grad():
                if hasattr(self.user_mappo.policy_net, 'ranking_weight'):
                    self.user_mappo.policy_net.ranking_weight.copy_(
                        torch.tensor(logit, dtype=self.user_mappo.policy_net.ranking_weight.dtype, device=self.user_mappo.policy_net.ranking_weight.device)
                    )
        except Exception:
            pass


