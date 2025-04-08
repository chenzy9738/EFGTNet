import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    # print('--------------sqrdists shape = ', sqrdists.shape)
    # print('--------------nsample = ', nsample)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)  # B, 1
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1  # 1, S
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False, knn=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint]
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)  # [B, npoint, C]
    torch.cuda.empty_cache()
    if knn:
        idx = knn_point(nsample, xyz, new_xyz)  # B x npoint x K
    else:
        idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PositionalEncodingFourier(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the
    "Attention is all of Need" paper. The implementation builds on DeTR code
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """

    def __init__(self, hidden_dim=64, dim=128, temperature=10000):
        super().__init__()
        self.token_projection = nn.Linear(hidden_dim * 3, dim)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim

    def forward(self, pos_embed, max_len=(1, 1, 1)):
        z_embed, y_embed, x_embed = pos_embed.chunk(3, 1)
        z_max, y_max, x_max = max_len

        eps = 1e-6
        z_embed = z_embed / (z_max + eps) * self.scale
        y_embed = y_embed / (y_max + eps) * self.scale
        x_embed = x_embed / (x_max + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=pos_embed.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed / dim_t
        pos_y = y_embed / dim_t
        pos_z = z_embed / dim_t

        pos_x = torch.stack((pos_x[:, 0::2].sin(),
                             pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(),
                             pos_y[:, 1::2].cos()), dim=2).flatten(1)
        pos_z = torch.stack((pos_z[:, 0::2].sin(),
                             pos_z[:, 1::2].cos()), dim=2).flatten(1)

        pos = torch.cat((pos_z, pos_y, pos_x), dim=1)

        pos = self.token_projection(pos)
        return pos


class LinformerEncoderLayer(nn.Module):

    def __init__(self, src_len, ratio, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout, inplace=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.linear_k = nn.Parameter(torch.empty(src_len // ratio, src_len))
        self.linear_v = nn.Parameter(torch.empty(src_len // ratio, src_len))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.dropout2 = nn.Dropout(dropout, inplace=False)

        self.activation = nn.ReLU(inplace=False)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_k)
        nn.init.xavier_uniform_(self.linear_v)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask=None, is_causal=False,
                src_key_padding_mask=None):  ######################## is_causal = False
        src_temp = src.transpose(0, 1)
        # print(src_temp.shape,self.linear_k.shape, self.linear_v.shape)
        key = torch.matmul(self.linear_k, src_temp).transpose(0, 1)
        value = torch.matmul(self.linear_v, src_temp).transpose(0, 1)
        src2 = self.self_attn(src, key, value, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class TransformerEncoderLayerPreNorm(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout, inplace=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.dropout2 = nn.Dropout(dropout, inplace=False)

        self.activation = nn.ReLU(inplace=False)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask=None, is_causal=False,
                src_key_padding_mask=None):  ###################### is_causal=  False
        src = self.norm1(src)
        src2, mask = self.self_attn(src, src, src, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)

        src = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)

        return src


class TransformerDecoderLayerPreNorm(nn.Module):

    def __init__(self, d_model, nc_mem, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout, inplace=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm_mem = nn.LayerNorm(nc_mem)

        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.dropout2 = nn.Dropout(dropout, inplace=False)
        self.dropout3 = nn.Dropout(dropout, inplace=False)

        self.activation = nn.ReLU(inplace=False)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt = self.norm1(tgt)
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt = self.norm2(tgt)
        memory = self.norm_mem(memory)
        tgt2, mask = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                         key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def lsh_knn(x, k, num_hashes=10):
    batch_size, num_dims, num_points = x.size()
    device = x.device

    # Create random projection matrices
    random_projections = torch.randn(num_hashes, num_dims, device=device)
    random_projections = F.normalize(random_projections, dim=1)

    # Project data to lower dimension
    projections = torch.matmul(x.transpose(2, 1), random_projections.t())

    # Compute hash buckets
    hash_buckets = torch.sign(projections)

    # Compute pairwise hamming distance
    pairwise_hamming_distance = torch.cdist(hash_buckets.float(), hash_buckets.float(), p=1)

    # Get the top k nearest neighbors
    idx = pairwise_hamming_distance.topk(k=k, dim=-1, largest=False)[1]
    return idx


import faiss
import numpy as np

def knn_faiss(x, k):
    batch_size, num_dims, num_points = x.size()
    device = x.device

    x_np = x.detach().cpu().numpy().astype('float32')  # Convert to NumPy array

    indices = []
    for i in range(batch_size):
        index = faiss.IndexFlatL2(num_dims)
        index.add(x_np[i].T)
        _, idx = index.search(x_np[i].T, k)
        indices.append(idx)

    idx = torch.tensor(np.stack(indices), device=device)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn_faiss(x, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)
        return total_loss


class EFFT(nn.Module):
    def __init__(self, npoint, nsample, dim_feature, dim_out, nhead=8, num_layers=4, drop=0.1):
        super(EFFT, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.nc_in = dim_feature
        self.nc_out = dim_out

        self.pe = nn.Sequential(
            nn.Conv2d(3, self.nc_in // 2, 1),
            nn.BatchNorm2d(self.nc_in // 2),
            nn.ReLU(),
            nn.Conv2d(self.nc_in // 2, self.nc_in, 1)
        )

        self.chunk = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.nc_in, nhead=nhead, dim_feedforward=2 * self.nc_in, dropout=drop),
            num_layers=num_layers)

        self.fc = nn.Conv2d(self.nc_in, self.nc_out, 1)

    def forward(self, xyz, features):
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        fps_idx = farthest_point_sample(xyz_flipped, self.npoint)
        new_xyz = index_points(xyz_flipped, fps_idx)
        group_idx = knn_point(self.nsample, xyz_flipped, new_xyz)
        grouped_xyz = index_points(xyz_flipped, group_idx).permute(0, 3, 1, 2).contiguous()
        grouped_features = index_points(features.transpose(1, 2).contiguous(), group_idx).permute(0, 3, 1,
                                                                                                  2).contiguous()

        position_encoding = self.pe(grouped_xyz)
        input_features = grouped_features + position_encoding
        B, D, np, ns = input_features.shape

        input_features = input_features.permute(0, 2, 1, 3).reshape(-1, D, ns).permute(2, 0, 1)  # (ns, B*np, D)
        transformed_feats = self.chunk(input_features).permute(1, 2, 0).reshape(B, np, D, ns).transpose(1, 2)
        output_features = F.max_pool2d(transformed_feats, kernel_size=[1, ns])
        output_features = self.fc(output_features).squeeze(-1)

        # Upsample to match original number of points
        output_features_upsampled = F.interpolate(output_features, size=(xyz.size(2)), mode='linear',
                                                  align_corners=True)

        return new_xyz.transpose(1, 2).contiguous(), output_features_upsampled


class EFET(nn.Module):
    def __init__(self, dim_feature, dim_out, nhead=8, num_layers=4, ratio=1, drop=0.1):
        super(EFET, self).__init__()

        self.nc_in = dim_feature
        self.nc_out = dim_out
        self.nhead = nhead
        self.src_pts = 4096 // ratio

        self.pe = nn.Sequential(
            nn.Conv2d(3, self.nc_in // 2, 1),
            nn.BatchNorm2d(self.nc_in // 2),
            nn.ReLU(),
            nn.Conv2d(self.nc_in // 2, self.nc_in, 1)
        )

        self.chunk = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.nc_in, nhead=nhead, dim_feedforward=2 * self.nc_in, dropout=drop),
            num_layers=num_layers)

        self.fc = nn.Conv2d(self.nc_in, self.nc_out, 1)

        self.linear_k = nn.Parameter(torch.empty(self.nc_in, self.nc_in))
        self.linear_v = nn.Parameter(torch.empty(self.nc_in, self.nc_in))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_k)
        nn.init.xavier_uniform_(self.linear_v)

    def forward(self, xyz, features):
        xyz_flipped = xyz.unsqueeze(-1)
        input_features = features.unsqueeze(-1) + self.pe(xyz_flipped)
        input_features = input_features.squeeze(-1).permute(2, 0, 1)

        B, C, N = input_features.shape
        src_temp = input_features.reshape(N, B * C)

        # Adjust the shape for matmul
        key = torch.matmul(self.linear_k, src_temp).reshape(N, B, C).permute(1, 2, 0)
        value = torch.matmul(self.linear_v, src_temp).reshape(N, B, C).permute(1, 2, 0)

        transformed_feats = self.chunk(input_features).permute(1, 2, 0)
        output_features = self.fc(transformed_feats.unsqueeze(-1)).squeeze(-1)

        return output_features


class RDGCNN(nn.Module):
    def __init__(self, k=40):
        super(RDGCNN, self).__init__()
        self.k = k
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv1 = nn.Sequential(
            nn.Conv2d(18, 32, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv5 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

    def forward(self, x):
        x = get_graph_feature(x, k=self.k)
        x1 = self.conv1(x).max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x2 = self.conv2(x).max(dim=-1, keepdim=False)[0] + x1

        x = get_graph_feature(x2, k=self.k)
        x3 = self.conv3(x).max(dim=-1, keepdim=False)[0] + x2

        x = get_graph_feature(x3, k=self.k)
        x4 = self.conv4(x).max(dim=-1, keepdim=False)[0] + x3

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        return x


class get_model(nn.Module):
    def __init__(self, num_classes=2):
        super(get_model, self).__init__()
        self.local_dgcnn = RDGCNN(k=60)
        self.local_transformer = EFFT(npoint=512, nsample=32, dim_feature=128, dim_out=128, drop=0.1, num_layers=4)
        self.global_transformer = EFET(dim_feature=256, dim_out=256, drop=0.1, ratio=4, num_layers=4)

        self.gcm_list = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(512, 64, 1),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv1d(512, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv1d(512, 256, 1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True)
            )
        ])
        self.gcm_end = nn.Sequential(
            nn.Conv1d(448, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        self.pred_head = nn.Sequential(
            nn.Conv1d(576, 256, 1),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Conv1d(256, num_classes, 1)
        )

    def forward(self, x):
        # Local feature extraction using RDGCNN
        x_local_dgcnn = self.local_dgcnn(x)

        # Local feature extraction using EFFT
        new_xyz, x_local_transformer = self.local_transformer(x[:, :3, :], x_local_dgcnn)

        # Concatenate RDGCNN and EFFT features
        x_local_combined = torch.cat([x_local_dgcnn, x_local_transformer], dim=1)

        # Global feature extraction using EFET
        x_global = self.global_transformer(x[:, :3, :], x_local_combined)

        # Concatenate RDGCNN, EFFT and EFET features
        x_combined = torch.cat([x_local_dgcnn, x_local_transformer, x_global], dim=1)

        # Global context mapping
        gcm_list = []
        for gcm in self.gcm_list:
            gcm_list.append(F.adaptive_max_pool1d(gcm(x_combined), 1))
        global_context = torch.cat(gcm_list, dim=1)
        global_context = self.gcm_end(global_context)

        # Prediction head
        global_context_repeated = global_context.repeat([1, 1, x_combined.shape[-1]])
        x_final = torch.cat([x_combined, global_context_repeated], dim=1)

        pred = self.pred_head(x_final)
        pred_o = pred.transpose(2, 1).contiguous()
        pred = F.log_softmax(pred_o, dim=-1)
        return pred, pred_o

