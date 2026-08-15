import torch
import torch.nn as nn


from pointnet_util import PointNetFeaturePropagation, PointNetSetAbstraction
from .transformer import TransformerBlock
import torch.nn.functional as F

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=24, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')
    # device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature      # (batch_size, 2*num_dims, num_points, k)




class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels, k_ratio=0.8):  # 多加了一个 k_ratio=0.8
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True, k_ratio=0.8)   # 多加了 k_ratio=k_ratio

    def forward(self, xyz, points,original_spectral):
        return self.sa(xyz, points,original_spectral)


class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])
    
    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1) #16 4 156
        feats2 = self.fc2(points2) #16 16 256
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2) # 16 16 256
        return feats1 + feats2
        

class Backbone(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim,
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        self.fc5 = nn.Sequential(
            nn.Linear(64, 32),###############
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = TransformerBlock(32, cfg.model.transformer_dim, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(32)
        self.bn6 = nn.BatchNorm1d(32)

        

        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
            self.transformers.append(TransformerBlock(channel, cfg.model.transformer_dim, nneighbor))
        self.nblocks = nblocks
        self.k = 3


        self.conv1 = nn.Sequential(nn.Conv2d(3*2, 32, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.pam1 = PointwiseAttentionModule(32, reduction_ratio=2)

        self.conv3 = nn.Sequential(nn.Conv2d(32*2, 32, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.pam2 = PointwiseAttentionModule(32, reduction_ratio=2)

        self.conv5 = nn.Sequential(nn.Conv2d(32*2, 32, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.pam3 = PointwiseAttentionModule(32, reduction_ratio=2)

        # self.conv6 = nn.Sequential(nn.Conv1d(112, 32, kernel_size=1, bias=False),
        #                            self.bn6,
        #                            nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(
            nn.Conv1d(112, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.15),  # 轻微正则
            nn.Conv1d(64, 32, kernel_size=1, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        batch_size, num_points, _ = x.shape
        xyz = x[..., :3]  # xyz:batch_size num_points coordinate;
        original_spectral = x[..., 3:8] if x.shape[-1] == 12 else x[..., 3:6]
        points = self.fc1(x)   # points:batch_size num_points 32;
 


        x1 = x1.transpose(2, 1)
        points = torch.cat((x1, points), dim=2)
        points = self.fc5(points)  # (batch_size, num_points, 64
        points = self.transformer1(xyz, points, original_spectral)[0] # batch_size num_points num_dim
        

        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points, original_spectral) # xyz:batch_size 512 3; points:batch_size 512 64
            points = self.transformers[i](xyz, points, original_spectral)[0]
            xyz_and_feats.append((xyz, points))
        return points, xyz_and_feats, DG, original_spectral


class PointTransformerCls(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Backbone(cfg)
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )
        self.nblocks = nblocks
    
    def forward(self, x):
        points, _ = self.backbone(x)
        res = self.fc2(points.mean(1))
        return res


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()

        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax_1 = nn.Softmax(dim=-1)
        self.softmax_2 = nn.Softmax(dim=-2)

    def forward(self, x):
        # 4 heads
        b, c, n = x.size() # b, c, n
        x_q = self.q_conv(x).permute(0, 2, 1)  # b, n, c
        x_q = torch.reshape(x_q, [b, 4, n, c // 4])  # b, 4, n, c/4

        x_k = self.k_conv(x)  # b, c, n
        x_k = torch.reshape(x_k, [b, 4, c // 4, n])  # b, 4, c/4, n
        x_v = self.v_conv(x)
        x_v = torch.reshape(x_v, [b, 4, c // 4, n])  # b, 4, c/4, n
        # b, n, n
        energy  = torch.matmul(x_q, x_k)  # b, 4, n, n
        # energy_shape: torch.Size([4, 4, 2048, 2048])

        attention = self.softmax_1(energy)
        attention = self.softmax_2(attention)  # b, 4, n, n
        # b, c, n
        x_r = torch.matmul(x_v, attention).reshape(b, c, n)  # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r # residual
        return x


class PointTransformerSeg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Backbone(cfg)
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 32 * 2 ** nblocks)
        )
        self.transformer2 = TransformerBlock(32 * 2 ** nblocks, cfg.model.transformer_dim, nneighbor)
        self.nblocks = nblocks
        self.nneighbor = nneighbor
        self.transition_ups = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in reversed(range(nblocks)):
            channel = 32 * 2 ** i
            self.transition_ups.append(TransitionUp(channel * 2, channel, channel))
            self.transformers.append(TransformerBlock(channel, cfg.model.transformer_dim, nneighbor))

        self.fc3 = nn.Sequential(
            nn.Linear(64, 128),###############
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )

        self.fctransformer = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )

        self.Multi_head = SA_Layer(channels=32)

        self.d_out0 = 64
        self.in_c2 = 32

        self.conv7 = nn.Conv2d(self.d_out0, self.in_c2, 1)
        self.conv8 = nn.Conv2d(self.in_c2*3, self.d_out0//2, 1)
        self.conv9 = nn.Conv2d(self.in_c2 * 3, self.d_out0 // 2, 1)
        self.conv10 = nn.Conv2d(int(3*self.d_out0/2), self.d_out0//2, 1)

        self.bn7 = nn.BatchNorm2d(self.in_c2)
        self.bn8 = nn.BatchNorm2d(self.d_out0//2)
        self.bn9 = nn.BatchNorm2d(self.d_out0 // 2)
        self.bn10 = nn.BatchNorm2d(self.d_out0 // 2)

    def knn_s(self, support_pts, query_pts, k):
        """
        :param support_pts: points you have, B*N1*3
        :param query_pts: points you want to know the neighbour index, B*N2*3
        :param k: Number of neighbours in knn search
        :return: neighbor_idx: neighboring points indexes, B*N2*k
        """

        B, N1, _ = support_pts.size()
        B, N2, _ = query_pts.size()

        s_pts = support_pts.unsqueeze(1).repeat(1, N2, 1, 1)
        q_pts = query_pts.unsqueeze(2).repeat(1, 1, N1, 1)

        distances = torch.sum((s_pts - q_pts) ** 2, dim=-1)
        neighbor_idx = torch.topk(distances, k, largest=False, dim=-1)[1]

        return neighbor_idx

    def gather_neighbour(self, pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        batch_size = pc.size(0)
        num_points = pc.size(1)
        d = pc.size(2)
        index_input = neighbor_idx.view(batch_size, -1)
        features = pc.gather(dim=1, index=index_input.unsqueeze(-1).expand(-1, -1, d))
        features = features.view(batch_size, num_points, neighbor_idx.size(-1), d)
        return features

    def forward(self, x):
        points, xyz_and_feats,DG, original_spectral  = self.backbone(x)
        xyz = xyz_and_feats[-1][0] # 16 4 3
        points = self.transformer2(xyz, self.fc2(points),original_spectral)[0] # points:16 4 512

        for i in range(self.nblocks):
            points = self.transition_ups[i](xyz, points, xyz_and_feats[- i - 2][0], xyz_and_feats[- i - 2][1])
            xyz = xyz_and_feats[- i - 2][0]
            points = self.transformers[i](xyz, points, original_spectral )[0]
        # return self.fctransformer(points) #transformer
        # points = torch.cat((DG, points), dim=2) #transformer+DGCNN 没有融合

        points = points.permute(0,2,1)
        points = self.Multi_head(points) #4,32,2048
        points = self.Multi_head(points)
        points = self.Multi_head(points)
        points = self.Multi_head(points)
        points = points.permute(0,2,1)
        feature = points # 2, 2048, 32

        Muti_DG = DG.permute(0,2,1)
        Muti_DG = self.Multi_head(Muti_DG) #4,32,2048
        Muti_DG = self.Multi_head(Muti_DG)
        Muti_DG = self.Multi_head(Muti_DG)
        Muti_DG = self.Multi_head(Muti_DG)
        Muti_DG = Muti_DG.permute(0,2,1)
        xyz_DG = Muti_DG #2, 2048, 32

        neigh_idx_xyz_DG = self.knn_s(xyz_DG, xyz_DG, self.nneighbor)
        neigh_idx_feature = self.knn_s(feature, feature, self.nneighbor)
        neigh_feat = self.gather_neighbour(feature.squeeze(2), neigh_idx_feature)  # B, N, k, d_out/2
        neigh_xyz = self.gather_neighbour(xyz_DG, neigh_idx_xyz_DG)  # B, N, k, in_c2
        tile_feat = feature.unsqueeze(2).repeat(1, 1, self.nneighbor, 1)  # B, N, k, d_out/2
        tile_xyz = xyz_DG.unsqueeze(2).repeat(1, 1, self.nneighbor, 1)  # B, N, k, in_c2

        feat_info = torch.cat([neigh_feat - tile_feat, tile_feat], dim=-1)  # B, N, k, d_out ************G(fj)
        feat_info = feat_info.permute(0,3,1,2)  # B, d_out, N, k ************G(fj)
        neigh_xyz_offsets = F.relu(self.bn7(self.conv7(feat_info)))  # B, in_c2, K, N->B, in_c2, N, K ***********M(G(fi))
        neigh_xyz = neigh_xyz.permute(0,3,1,2)
        shifted_neigh_xyz = neigh_xyz + neigh_xyz_offsets  # B, N, k, in_c2 ->B, in_c2, N, K***************~pi

        tile_xyz = tile_xyz.permute(0,3,1,2) # B, in_c2, N, k
        xyz_info = torch.cat([neigh_xyz - tile_xyz, shifted_neigh_xyz, tile_xyz], dim=1)  # B, N, k, 9 -> B, in_c2*3, N, k*** ~G(pi)
        neigh_feat_offsets = F.relu(self.bn8(self.conv8(xyz_info)))  # B, d_out/2, k, N *******M(~G(pi))=xyz_encoding
        neigh_feat = neigh_feat.permute(0,3,1,2)
        shifted_neigh_feat = neigh_feat + neigh_feat_offsets  # B, N, k, d_out2 ->B, d_out2, N, k *****************~fi = M(~G(pi)) + fj

        xyz_encoding = F.relu(self.bn9(self.conv9(xyz_info)))  # B, d_out2, k, N -> B, d_out2, N, k *********************M(~G(pi))
        feat_info = torch.cat([shifted_neigh_feat, feat_info], dim=1)  # B, N, k, d_out -> B, d_out, N, k ************~G(fj) = fi + ~fj +fj-fi
        feat_encoding = F.relu(self.bn10(self.conv10(feat_info)))  # B, d_out/2, k, N -> B, d_out/2, N , k***********M(~G(fj))

        # Mixed Local Aggregation
        overall_info = torch.cat([xyz_encoding, feat_encoding], dim=1)  # B, N, k, d_out -> B, d_out, N, k
        # overall_info_shape: torch.Size([4, 64, 2048]) ->B, d_out, N
        overall_info = overall_info.max(dim=-1, keepdim=False)[0]
       
        overall_info = overall_info.permute(0,2,1) # overall_info_shape: torch.Size([4, 2048, 64]) ->B, N，d_out
        D_out = self.fc3(overall_info)


        return D_out