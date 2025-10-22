import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalTimeAwareEncoder(nn.Module):
    def __init__(self, user_num, cate_size, embed_dim, hidden_dim, poi_num):
        super(HierarchicalTimeAwareEncoder, self).__init__()
        self.user_num = user_num
        self.poi_num = poi_num
        self.embed_dim = embed_dim

        self.user_embedding = nn.Embedding(user_num + 1, embed_dim)
        self.poi_embedding = nn.Embedding(poi_num + 1, embed_dim)
        self.cat_embedding = nn.Embedding(cate_size, embed_dim)
        self.hour_embedding = nn.Embedding(24, embed_dim)
        self.week_embedding = nn.Embedding(7, embed_dim)

        # 
        self.feature_fusion_layer = nn.Linear(embed_dim * 4, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=4, 
            dim_feedforward=hidden_dim * 2, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.positional_encoding = nn.Embedding(50, hidden_dim) # 假设最大序列长度为50

        self.dropout = nn.Dropout(0.2)
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def forward(self, hour, week, cat, hour_pre, week_pre, x, userid):
        # --- FIX: Add robust clamping to prevent all index-out-of-bounds errors ---
        userid = userid.long().clamp(min=0, max=self.user_embedding.num_embeddings - 1)
        x = x.long().clamp(min=0, max=self.poi_embedding.num_embeddings - 1)
        cat = cat.long().clamp(min=0, max=self.cat_embedding.num_embeddings - 1)
        hour = hour.long().clamp(min=0, max=self.hour_embedding.num_embeddings - 1)
        week = week.long().clamp(min=0, max=self.week_embedding.num_embeddings - 1)
        hour_pre = hour_pre.long().clamp(min=0, max=self.hour_embedding.num_embeddings - 1)
        week_pre = week_pre.long().clamp(min=0, max=self.week_embedding.num_embeddings - 1)
        # --- END OF FIX ---

        batch_size = x.shape[0]
        seq_len = x.shape[1]

        user_embed = self.user_embedding(userid)
        poi_embed = self.poi_embedding(x)
        cat_embed = self.cat_embedding(cat)
        hour_embed = self.hour_embedding(hour)
        week_embed = self.week_embedding(week)

        # 确保所有嵌入向量的序列长度一致
        min_seq_len = min(poi_embed.shape[1], cat_embed.shape[1], hour_embed.shape[1], week_embed.shape[1])
        poi_embed = poi_embed[:, :min_seq_len, :]
        cat_embed = cat_embed[:, :min_seq_len, :]
        hour_embed = hour_embed[:, :min_seq_len, :]
        week_embed = week_embed[:, :min_seq_len, :]

        # Transformer-based 特征融合与序列编码
        # 1. 特征融合
        fused_features = torch.cat([poi_embed, cat_embed, hour_embed, week_embed], dim=-1)
        fused_features = F.relu(self.feature_fusion_layer(fused_features))

        # 2. 添加位置编码
        positions = torch.arange(0, min_seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        pos_embed = self.positional_encoding(positions)
        fused_features = fused_features + pos_embed 

        # 3. Transformer编码
        transformer_output = self.transformer_encoder(fused_features)

        # 4. 序列信息聚合 (使用平均池化)
        sequence_representation = transformer_output.mean(dim=1)

        # 5. 结合用户偏好
        final_feature = sequence_representation + user_embed.squeeze(1)

        return final_feature, None
