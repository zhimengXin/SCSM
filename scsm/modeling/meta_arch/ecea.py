import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

class SinePositionalEncoding(nn.Module):
    """Sine-based positional encoding used in DETR and Deformable DETR."""
    
    def __init__(self, num_feats=128, temperature=10000, normalize=True):
        super().__init__()
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        
    def forward(self, x):
        """x: tensor of shape (B, H, W) or (B, C, H, W)"""
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
        
        not_mask = torch.ones_like(x[:, 0, :, :])  # (B, H, W)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi
        
        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)
        
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), 
                            pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), 
                            pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def build_positional_encoding(pos_encoding_cfg):
    """Factory function for positional encodings."""
    if pos_encoding_cfg['type'] == 'SinePositionalEncoding':
        return SinePositionalEncoding(
            num_feats=pos_encoding_cfg['num_feats'],
            normalize=pos_encoding_cfg['normalize']
        )
    else:
        raise ValueError(f"Unknown positional encoding type: {pos_encoding_cfg['type']}")


class ExtensibleAttention_single(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)       
        
        self.pos_proj = nn.Linear(embed_dim, num_heads * self.head_dim)
        self.offset_proj = nn.Linear(embed_dim, num_heads * 2)       
        
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        nn.init.zeros_(self.offset_proj.weight)
        nn.init.zeros_(self.offset_proj.bias)
        
    def forward(self, query, key, value, reference_points, pos_embed=None):
        N, L, C = query.shape
        H = self.num_heads
        
        q = self.q_proj(query).view(N, L, H, self.head_dim)
        k = self.k_proj(key).view(N, L, H, self.head_dim)
        v = self.v_proj(value).view(N, L, H, self.head_dim)
        
        # pos encoding
        if pos_embed is not None:
            pos_embed = self.pos_proj(pos_embed).view(N, L, H, self.head_dim)
            q = q + pos_embed
            k = k + pos_embed
        
        
        #  single point
        offsets = self.offset_proj(query)
        offsets = offsets.view(N, L, H, 2)
        
        sampling_points = reference_points.unsqueeze(2) + offsets

       
        
        attn_weights = torch.zeros(N, L, H, device=query.device)
        output = torch.zeros(N, L, H, self.head_dim, device=query.device)
        
        for i in range(L):
            q_i = q[:, i]
            k_sampled = self.sample_pointssingle(k, sampling_points[:, i])
            a = torch.einsum('nhd,nhd->nh', q_i, k_sampled) / (self.head_dim ** 0.5)
            if a.shape == attn_weights[:, i].shape:
                attn_weights[:, i] = a
            else:
                b = 1
            
            v_sampled = self.sample_pointssingle(v, sampling_points[:, i])
            output[:, i] = torch.einsum('nh,nhd->nhd', F.softmax(attn_weights[:, i], dim=-1), v_sampled)
        
        output = output.view(N, L, C)
        output = self.out_proj(output)
        output = self.dropout(output)
        
        return output
    
    def sample_pointssingle(self, x, points):
        
        N, L, H, K, _ = points.shape
        D = self.head_dim
        
        
        points = points * 2 - 1  
        points = points.view(N*L*H*K, 2)
        
        
        x = x.permute(0, 1, 3, 2).reshape(N*L, H, D, 1)
        
       
        sampled = F.grid_sample(
            x, 
            points.unsqueeze(1).unsqueeze(1),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        
        return sampled.view(N, L, H, K, D)



class ExtensibleAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_points=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads
        
        # project
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # extensible point prediction（multi points）
        self.offset_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, num_heads * num_points * 2)
        )
        
        
        self.pos_proj = nn.Linear(embed_dim, num_heads * self.head_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        
        nn.init.zeros_(self.offset_proj[-1].weight)
        nn.init.zeros_(self.offset_proj[-1].bias)
        self._reset_parameters()

    def _reset_parameters(self):
       
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.offset_proj[-1].bias, bias_value)

    def forward(self, query, key, value, reference_points, pos_embed=None):
        N, L, C = query.shape
        H, K = self.num_heads, self.num_points
        
       
        q = self.q_proj(query).view(N, L, H, self.head_dim)
        k = self.k_proj(key).view(N, L, H, self.head_dim)
        v = self.v_proj(value).view(N, L, H, self.head_dim)
        
        # 处理位置编码
        if pos_embed is not None:
            pos_embed = self.pos_proj(pos_embed).view(N, L, H, self.head_dim)
            q = q + pos_embed
            k = k + pos_embed
        
        #  (N, L, H*K*2)
        offsets = self.offset_proj(query).view(N, L, H, K, 2)
        
        # extensible points (N, L, H, K, 2)
        ref_points = reference_points.unsqueeze(2).unsqueeze(2)  # (N, L, 1, 1, 2)
        ref_points = ref_points.repeat(1, 1, 1, K, 1)  # 复制K次
        sampling_points = ref_points + offsets
        
        
        sampled_k = self.sample_points(k, sampling_points)  # (N, L, H, K, D)
        sampled_v = self.sample_points(v, sampling_points)  # (N, L, H, K, D)
        
        # extensible attention
        attn_weights = torch.einsum('nlhd,nlhkd->nlhk', q, sampled_k) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)  # (N, L, H, K)
        
       
        output = torch.einsum('nlhk,nlhkd->nlhd', attn_weights, sampled_v)
        output = output.reshape(N, L, C)
        
        return self.out_proj(self.dropout(output))
    
    
    
    def sample_points(self, x, points):        
        N, H, W, C = x.shape          # input shape [N,H,W,C]
        _, _, _, K, _ = points.shape  # net shape [N,H,W,K,2]
        
        #  [N,H,W,C] -> [N*H*W, C, 1, 1]
        x = x.reshape(N*H*W, C, 1, 1)
        
        #  [N*H*W, C, 1, 1] -> [N*H*W*K, C, 1, 1]
        x = x.repeat_interleave(K, dim=0)
        
        # =[N,H,W,K,2] -> [N*H*W*K, 1, 1, 2] and nomal
        points = points.reshape(N*H*W*K, 1, 1, 2) * 2 - 1
        
        # batch sample
        sampled = F.grid_sample(
            x, 
            points,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )  # ouput shape [N*H*W*K, C, 1, 1]
        
        # restore shape [N,H,W,K,C]
        return sampled.reshape(N, H, W, K, C)

class ExtensibleTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_points=4, dropout=0.1):
        super().__init__()
        self.self_attn = ExtensibleAttention(embed_dim, num_heads, num_points, dropout)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src, pos_embed, reference_points):
        # 修改为将pos_embed直接传递给ExtensibleAttention
        src2 = self.self_attn(src, src, src, reference_points, pos_embed)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(F.relu(self.linear1(src)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class ECEA(nn.Module):
    def __init__(self, backbone, num_classes=80, num_queries=100, embed_dim=256, num_heads=8, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        self.backbone = backbone
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        
        self.input_proj = nn.Conv2d(backbone.num_channels, embed_dim, kernel_size=1)
        
       
        self.pos_embed = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        )
        
        self.encoder_layers = nn.ModuleList([
            ExtensibleTransformerEncoderLayer(embed_dim, num_heads) 
            for _ in range(num_encoder_layers)
        ])
        
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(embed_dim, num_heads) 
            for _ in range(num_decoder_layers)
        ])
        
        self.class_embed = nn.Linear(embed_dim, num_classes + 1)
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)
        
    def forward(self, x):
        features = self.backbone(x)
        src = self.input_proj(features[-1])
        
        N, C, H, W = src.shape
        
        
        pos_embed = self.pos_embed(torch.zeros_like(src))
        pos_embed = pos_embed.flatten(2).permute(0, 2, 1)
        
        reference_points = self.generate_reference_points(H, W, device=x.device)
        
        src = src.flatten(2).permute(0, 2, 1)
        reference_points = reference_points.unsqueeze(0).repeat(N, 1, 1)
        
        for layer in self.encoder_layers:
            src = layer(src, pos_embed, reference_points)
        
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(N, 1, 1)
        tgt = torch.zeros_like(query_embed)
        for layer in self.decoder_layers:
            tgt = layer(tgt, src)
        
        outputs_class = self.class_embed(tgt)
        outputs_coord = self.bbox_embed(tgt).sigmoid()
        
        return {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
    
    def generate_reference_points(self, H, W, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device)
        )
        ref_y = ref_y.reshape(-1) / H
        ref_x = ref_x.reshape(-1) / W
        return torch.stack((ref_x, ref_y), dim=1)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
