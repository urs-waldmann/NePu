'''
Implementation of the Neural Puppeteer architecture
Author: Simon Giebenhain (simon.giebenhain@uni.kn)
'''

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.nn.functional as F

from .geometry import plucker_embedding




def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


class TransformerBlock(nn.Module):
    """
    Attributes:
        d_model (int): number of input, output and internal dimensions
        k (int): number of points among which local attention is calculated
        shift (bool): Whether model should learn to translate the points
        group_all (bool): When true full instead of local attention is calculated
    """
    def __init__(self, d_model, k, d_space, group_all=False):
        super().__init__()

        self.bn = nn.BatchNorm1d(d_model)

        d_bottle = d_model

        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_bottle),
            nn.ReLU(),
            nn.Linear(d_bottle, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)

        self.k = k
        self.group_all = group_all

        N_freq = 5
        self.freq_bands = torch.linspace(1, 2.**N_freq, steps=N_freq)

        self.sum_pos_embed = nn.Linear(d_space*2*N_freq + d_space, d_model)


    def forward(self, xyz, feats=None):
        """
        :param xyz [b x n x 3]: positions in point cloud
        :param feats [b x n x d]: features in point cloud
        :return:
            new_features [b x n x d]:
        """

        with torch.no_grad():
            # full attention
            if self.group_all:
                b, n, _ = xyz.shape
                knn_idx = torch.arange(n, device=xyz.device).unsqueeze(0).unsqueeze(1).repeat(b, n, 1)
            # local attention
            else:
                # kNN grouping
                dists = square_distance(xyz, xyz)
                k = min(self.k, dists.shape[2]) #TODO: achtung
                knn_idx = dists.argsort()[:, :, :k]  # b x n x k

            knn_xyz = index_points(xyz, knn_idx)

        ori_feats = feats
        x = feats

        q_attn = self.w_qs(x)
        k_attn = index_points(self.w_ks(x), knn_idx)
        v_attn = index_points(self.w_vs(x), knn_idx)


        pos_diff = xyz[:, :, None] - knn_xyz  # torch.norm(xyz[:, :, None] - knn_xyz, dim=-1, keepdim=True)
        pos_embeds = [pos_diff]
        for freq in self.freq_bands:
            pos_embeds.append(torch.sin(pos_diff * freq))
            pos_embeds.append(torch.cos(pos_diff * freq))

        pos_embed = torch.cat(pos_embeds, dim=-1)
        pos_encode = self.sum_pos_embed(pos_embed)

        edges = q_attn[:, :, None] - k_attn + pos_encode
        attn = self.fc_gamma(edges)

        attn = functional.softmax(attn, dim=-2)  # b x n x k x d
        res = torch.einsum('bmnf,bmnf->bmf', attn, v_attn + pos_encode)

        res = res + ori_feats

        res = self.bn(res.permute(0, 2, 1)).permute(0, 2, 1)

        return res


class CrossTransformerBlock(nn.Module):
    def __init__(self, dim_inp, dim, dim_space, nneigh=7, reduce_dim=True):
        super().__init__()

        self.dim = dim

        self.nneigh = nneigh

        d_bottle = dim

        self.fc_gamma = nn.Sequential(
            nn.Linear(dim, d_bottle),
            nn.ReLU(),
            nn.Linear(d_bottle, dim)
        )
        self.w_k_global = nn.Linear(dim_inp, dim, bias=False)
        self.w_v_global = nn.Linear(dim_inp, dim, bias=False)


        self.w_qs = nn.Linear(dim_inp, dim, bias=False)
        self.w_ks = nn.Linear(dim_inp, dim, bias=False)
        self.w_vs = nn.Linear(dim_inp, dim, bias=False)

        if not reduce_dim:
            self.fc = nn.Linear(dim, dim_inp)
        self.reduce_dim = reduce_dim

        N_freq = 5
        self.freq_bands = torch.linspace(1, 2.**N_freq, steps=N_freq)

        self.sum_pos_embed = nn.Linear(3*2*N_freq + 3, dim)

    # xyz_q: B x n_queries x 3
    # lat_rep: B x dim
    # xyz: B x n_anchors x 3,
    # points: B x n_anchors x dim
    def forward(self, xyz_q, lat_rep, xyz, points):
        with torch.no_grad():
            dists = square_distance(xyz_q, xyz[:, :, :2])
            ## knn group
            if self.nneigh == -1:
                knn_idx = dists.argsort()
            else:
                knn_idx = dists.argsort()[:, :, :self.nneigh]  # b x nQ x k
            #knn_idx = pointops.knnquery_heap(self.nneigh, xyz, xyz_q).long()

        b, nQ, _ = xyz_q.shape

        if len(lat_rep.shape) == 2:
            q_attn = self.w_qs(lat_rep).unsqueeze(1).repeat(1, nQ, 1)
            k_global = self.w_k_global(lat_rep).unsqueeze(1).repeat(1, nQ, 1).unsqueeze(2)
            v_global = self.w_v_global(lat_rep).unsqueeze(1).repeat(1, nQ, 1).unsqueeze(2)

        else:
            q_attn = self.w_qs(lat_rep)
            k_global = self.w_k_global(lat_rep).unsqueeze(2)
            v_global = self.w_v_global(lat_rep).unsqueeze(2)

        k_attn = index_points(self.w_ks(points),
                              knn_idx)  # b, nQ, k, dim
        k_attn = torch.cat([k_attn, k_global], dim=2)
        v_attn = index_points(self.w_vs(points), knn_idx)  # #self.w_vs(points).unsqueeze(1).repeat(1, nQ, 1, 1)
        v_attn = torch.cat([v_attn, v_global], dim=2)
        xyz = index_points(xyz, knn_idx)  # xyz = xyz.unsqueeze(1).repeat(1, nQ, 1, 1)
        #pos_encode = self.fc_delta(torch.cat([xyz_q[:, :, None] - xyz[:, :, :, :2], xyz[:, :, :, -1:]], dim=-1))  # b x nQ x k x dim
        pos_diff = torch.cat([xyz_q[:, :, None] - xyz[:, :, :, :2], xyz[:, :, :, -1:]], dim=-1)
        pos_embeds = [pos_diff]
        for freq in self.freq_bands:
            pos_embeds.append(torch.sin(pos_diff * freq))
            pos_embeds.append(torch.cos(pos_diff * freq))

        pos_embed = torch.cat(pos_embeds, dim=-1)
        pos_encode = self.sum_pos_embed(pos_embed)
        pos_encode = torch.cat([pos_encode, torch.zeros([b, nQ, 1, self.dim], device=pos_encode.device)],
                               dim=2)  # b, nQ, k+1, dim

        attn = self.fc_gamma(q_attn[:, :, None] - k_attn + pos_encode)
        #res = torch.max(attn + pos_encode, dim=-2)[0]  # b x nQ x dim

        attn = functional.softmax(attn, dim=-2)  # b x n x k x d
        res = torch.einsum('bmnf,bmnf->bmf', attn, v_attn + pos_encode)


        if not self.reduce_dim:
            res = self.fc(res)
        return res


class ElementwiseMLP(nn.Module):
    """
    Simple MLP, consisting of two linear layers, a skip connection and batch norm.
    More specifically: linear -> BN -> ReLU -> linear -> BN -> ReLU -> resCon -> BN
    """
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, 1)
        self.conv2 = nn.Conv1d(dim, dim, 1)
        self.bn = nn.BatchNorm1d(dim)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        """
        :param x: [B x n x d]
        :return: [B x n x d]
        """
        x = x.permute(0, 2, 1)
        return self.bn(x + self.ReLU(self.conv2(self.ReLU(self.conv1(x))))).permute(0, 2, 1)


class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.
    Copied from https://github.com/autonomousvision/convolutional_occupancy_networks

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class NePuEncoder(nn.Module):
    """
    AIR-Net encoder.

    Attributes:
        npoints_per_layer [int]: cardinalities of point clouds for each layer
        nneighbor int: number of neighbors in local vector attention (also in TransformerSetAbstraction)
        nneighbor_reduced int: number of neighbors in very first TransformerBlock
        nfinal_transformers int: number of full attention layers
        d_transformer int: dimensionality of model
        d_reduced int: dimensionality of very first layers
        full_SA bool: if False, local self attention is used in final layers
        shift bool: if True, models learns to deform the points in the point cloud
        has_features bool: True, when input has signed-distance value for each point
    """
    def __init__(self, nfinal_transformers, d_transformer, d_space, lat_dim, feature_dim):
        super().__init__()
        self.d_transformer = d_transformer
        self.d_space = d_space
        self.feature_dim = feature_dim

        self.fc_final = nn.Sequential(
            nn.Linear(d_transformer, lat_dim),
            nn.ReLU(),
            nn.Linear(lat_dim, lat_dim)
        )
        #TODO: noBottle: self.fc_feats = nn.Linear(d_transformer, 200)

        self.enc_feats = nn.Linear(self.feature_dim, d_transformer)

        self.transformer_begin = TransformerBlock(d_transformer, -1, self.d_space, group_all=True)

        self.final_transformers = nn.ModuleList()
        self.final_elementwise = nn.ModuleList()

        for i in range(nfinal_transformers):
            self.final_transformers.append(
                TransformerBlock(d_transformer, -1, self.d_space, group_all=True)
            )
        for i in range(nfinal_transformers):
            self.final_elementwise.append(
                ElementwiseMLP(dim=d_transformer)
            )



    def forward(self, xyz, feats):
        """
        :param xyz [B x n x 3] (or [B x n x 4], but then has_features=True): input point cloud
        :param intermediate_out_path: path to store point cloud after every deformation to
        :return: global latent representation [b x d_transformer]
                 xyz [B x npoints_per_layer[-1] x d_transformer]: anchor positions
                 feats [B x npoints_per_layer[-1] x d_transformer: local latent vectors
        """

        feats = self.enc_feats(feats)
        xyz = xyz.contiguous()
        feats = self.transformer_begin(xyz * 4, feats)

        for i, att_block in enumerate(self.final_transformers):
            feats = att_block(xyz * 4, feats)
            feats = self.final_elementwise[i](feats)

        embd = self.fc_final(feats)

        # max pooling
        lat_rep = embd.max(dim=1)[0]

        return lat_rep# TODO: noBottle, self.fc_feats(feats)


class RenderHead(nn.Module):
    """
    Rendering Head, responsible for prediction of single modality.

    Attributes:

    """
    def __init__(self, loc_lat_dim, out_dim, hidden_dim=64, n_blocks=5):
        super(RenderHead, self).__init__()
        self.init_enc = nn.Linear(loc_lat_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_dim) for i in range(n_blocks)
        ])

        self.fc_c = nn.ModuleList([
            nn.Linear(loc_lat_dim, hidden_dim) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.actvn = F.relu
        self.n_blocks = n_blocks

    def forward(self, y):
        net = self.init_enc(y)

        for i in range(self.n_blocks):
            net = net + self.fc_c[i](y)
            net = self.blocks[i](net)

        return self.fc_out(self.actvn(net))

#TODO resturcture heads
class NePuRenderer(nn.Module):
    """
    NePu Renderer

    Attributes:

    """
    def __init__(self, lat_dim, loc_lat_dim, dim_space, nneigh=7, hidden_dim=64, n_blocks=5):
        super().__init__()
        self.dim_space = dim_space
        self.n_blocks = n_blocks

        self.down = nn.Linear(lat_dim, loc_lat_dim)

        self.ct1 = CrossTransformerBlock(loc_lat_dim, loc_lat_dim, None, nneigh=nneigh)

        self.occ_head   = RenderHead(loc_lat_dim=loc_lat_dim, out_dim=1, hidden_dim=hidden_dim, n_blocks=n_blocks)
        self.depth_head = RenderHead(loc_lat_dim=loc_lat_dim, out_dim=1, hidden_dim=hidden_dim, n_blocks=n_blocks)
        self.color_head = RenderHead(loc_lat_dim=loc_lat_dim, out_dim=3, hidden_dim=hidden_dim, n_blocks=n_blocks)


        self.trans2d_1 = TransformerBlock(loc_lat_dim, 8, 3)
        self.trans2d_2 = TransformerBlock(loc_lat_dim, -1, 3, group_all=True)
        self.trans2d_3 = TransformerBlock(loc_lat_dim, -1, 3, group_all=True)



    def forward(self, xyz_q, encoding, camera_params, kps_2d):
        """
        TODO update commont to include encoding dict
        :param xyz_q [B x n_queries x 3]: queried 3D coordinates
        :param lat_rep [B x dim_inp]: global latent vectors
        :param xyz [B x n_anchors x 3]: anchor positions
        :param feats [B x n_anchros x dim_inp]: local latent vectors
        :return: occ [B x n_queries]: occupancy probability for each queried 3D coordinate
        """

        lat_rep = self.down(encoding['z'])
        xyz = encoding['anchors']
        feats = encoding['anchor_feats']

        xyz = xyz.contiguous()

        keyPos_hom = torch.cat([xyz, torch.ones([xyz.shape[0], xyz.shape[1], 1], device=xyz.device, dtype=torch.float)], dim=2).permute(0, 2, 1)
        tmp = torch.bmm(camera_params['extrinsics'], keyPos_hom).permute(0, 2, 1)
        d = tmp[:, :, -1].unsqueeze(-1)


        min_d = torch.min(d, dim=1)[0].unsqueeze(1)
        d -= min_d
        d = d / torch.max(d, dim=1)[0].unsqueeze(1)
        kps_2dd = torch.cat([kps_2d, d], dim=-1)

        feats = self.trans2d_1(kps_2dd, feats)
        feats = self.trans2d_2(kps_2dd, feats)
        feats = self.trans2d_3(kps_2dd, feats)

        lat_rep = self.ct1(xyz_q, lat_rep, kps_2dd, feats)

        occ = self.occ_head(lat_rep)
        depth = self.depth_head(lat_rep)
        color = self.color_head(lat_rep)

        return occ, depth, color


class GlobalLFNDecoder(nn.Module):
    def __init__(self, dim, dim_space,  hidden_dim=64, n_blocks=5):
        super().__init__()
        self.dim_space = dim_space
        self.n_blocks = n_blocks

        self.init_enc = nn.Linear(dim + 128, hidden_dim)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_dim) for i in range(n_blocks)
        ])

        self.fc_c = nn.ModuleList([
            nn.Linear(dim + 128, hidden_dim) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_dim, 1)

        self.init_enc_depth = nn.Linear(dim + 128, hidden_dim)
        #
        self.blocks_depth = nn.ModuleList([
            ResnetBlockFC(hidden_dim) for i in range(n_blocks)
        ])
        #
        self.fc_c_depth = nn.ModuleList([
            nn.Linear(dim + 128, hidden_dim) for i in range(n_blocks)
        ])
        #
        self.fc_out_depth = nn.Linear(hidden_dim, 1)

        self.init_enc_color = nn.Linear(dim + 128, hidden_dim)
        #
        self.blocks_color = nn.ModuleList([
            ResnetBlockFC(hidden_dim) for i in range(n_blocks)
        ])
        #
        self.fc_c_color = nn.ModuleList([
            nn.Linear(dim + 128, hidden_dim) for i in range(n_blocks)
        ])
        #
        self.fc_out_color = nn.Linear(hidden_dim, 3)

        self.actvn = F.relu

        N_freq = 5
        self.freq_bands = torch.linspace(1, 2.**N_freq, steps=N_freq)
        self.sum_pos_embed = nn.Linear(6*2*N_freq + 6, 128)




    def forward(self, xyz_q, encoding, camera_params, kps_2d):
        """
        TODO update commont to include encoding dict
        :param xyz_q [B x n_queries x 3]: queried 3D coordinates
        :param lat_rep [B x dim_inp]: global latent vectors
        :param xyz [B x n_anchors x 3]: anchor positions
        :param feats [B x n_anchros x dim_inp]: local latent vectors
        :return: occ [B x n_queries]: occupancy probability for each queried 3D coordinate
        """

        lat_rep = encoding['z'].unsqueeze(1).repeat(1, xyz_q.shape[1], 1)
        #trans = camera_params['trans'] # b x 3
        #trans = trans.unsqueeze(1).repeat(1, xyz_q.shape[1], 1)

        #pos_info = torch.cat([trans, xyz_q], dim=-1)
        eyE = torch.eye(4, device=xyz_q.device).unsqueeze(0).repeat(xyz_q.shape[0], 1, 1)
        eyI = torch.eye(4, device=xyz_q.device).unsqueeze(0).repeat(xyz_q.shape[0], 1, 1)
        eyE[:, :3, :4] = camera_params['ex_inv']
        eyI[:, :3, :3] = camera_params['intrinsics']
        pos_info = plucker_embedding(eyE, xyz_q.flip(2), eyI)
        pos_embeds = [pos_info]
        for freq in self.freq_bands:
            pos_embeds.append(torch.sin(pos_info * freq))
            pos_embeds.append(torch.cos(pos_info * freq))

        pos_embed = torch.cat(pos_embeds, dim=-1)
        pos_encode = self.sum_pos_embed(pos_embed)
        lat_rep = torch.cat([lat_rep, pos_encode], dim=-1)


        net = self.init_enc(lat_rep)

        for i in range(self.n_blocks):
            net = net + self.fc_c[i](lat_rep)
            net = self.blocks[i](net)

        occ = self.fc_out(self.actvn(net))

        net_depth = self.init_enc_depth(lat_rep)
        #
        for i in range(self.n_blocks):
            net_depth = net_depth + self.fc_c_depth[i](lat_rep)
            net_depth = self.blocks_depth[i](net_depth)
        #
        depth = self.fc_out_depth(self.actvn(net_depth))

        net_color = self.init_enc_color(lat_rep)
        #
        for i in range(self.n_blocks):
            net_color = net_color + self.fc_c_color[i](lat_rep)
            net_color = self.blocks_color[i](net_color)
        #
        color = self.fc_out_color(self.actvn(net_color))

        return occ, depth, color


class DecoderDummy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z):

        return {'z': z}


class NePuDecoder(nn.Module):

    def __init__(self, d_model, nkps, loc_lat_dim, d_kps_bottle):
        super().__init__()
        #TODO for color512-07: no bottleneck at all and AIRenc
        self.mlp_pos = nn.Sequential(
            nn.Linear(d_model, d_kps_bottle),
            nn.ReLU(),
            nn.Linear(d_kps_bottle, d_model),  # int(d_model/2)),#TODO undo
            nn.ReLU(),
            nn.Linear(d_model, nkps*3)  #int(d_model/2), nkps*3) #TODO undo half
        )

        self.mlp_feats = nn.Sequential(
            nn.Linear(d_model, 2*d_model),
            nn.ReLU(),
            nn.Linear(2*d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, nkps*loc_lat_dim)
        )
        self.t1 = TransformerBlock(loc_lat_dim, 8, 3, group_all=False)
        #self.t2 = TransformerBlock(200, 8, 3, group_all=False)
        self.t3 = TransformerBlock(loc_lat_dim, -1, 3, group_all=True)
        self.t4 = TransformerBlock(loc_lat_dim, -1, 3, group_all=True)

        self.nkps = nkps
        self.loc_lat_dim = loc_lat_dim



    def forward(self, z):
        pos_init = self.mlp_pos(z).view(-1, self.nkps, 3)
        feats = self.mlp_feats(z).view(-1, self.nkps, self.loc_lat_dim)
        pos = pos_init * 4
        feats = self.t1(pos, feats)
        #feats = self.t2(pos, feats)
        feats = self.t3(pos, feats)
        feats = self.t4(pos, feats)
        pos = pos / 4
        return {'z': z, 'anchors': pos, 'anchor_feats': feats}


def get_encoder(CFG):
    CFG_enc = CFG['encoder']
    encoder = NePuEncoder(nfinal_transformers=CFG_enc['nfinal_trans'],
                            d_transformer=CFG_enc['encoder_attn_dim'],
                            d_space=3,
                            lat_dim=CFG_enc.get('lat_dim', 512),
                            feature_dim=CFG['data']['nkps'])
    return encoder


def get_decoder(CFG):
    CFG_enc = CFG['encoder']
    dim = CFG_enc.get('lat_dim', 512)
    d_kps_bottle = CFG_enc.get('kps_bottle', int(dim/2))
    loc_lat_dim = CFG_enc.get('loc_lat_dim', 200)
    if CFG['renderer']['type'] == 'lfn':
        decoder = DecoderDummy()
    else:
        decoder = NePuDecoder(d_model=dim,
                                nkps=CFG['data']['nkps'],
                                loc_lat_dim=loc_lat_dim,
                                d_kps_bottle=d_kps_bottle)
    return decoder


def get_renderer(CFG):
    CFG_enc = CFG['encoder']
    CFG_render = CFG['renderer']
    if CFG_render['type'] == 'nepu': #TODO improve cfg files
        decoder = NePuRenderer(lat_dim=CFG_enc['lat_dim'],
                                loc_lat_dim=CFG_enc['loc_lat_dim'],
                                dim_space=2,
                                nneigh=CFG_render['decoder_nneigh'],
                                hidden_dim=CFG_render['decoder_hidden_dim'])
    elif CFG_render['type'] == 'lfn':
        decoder = GlobalLFNDecoder(CFG_enc['lat_dim'],
                                -1, #TODO dim space
                                hidden_dim=CFG_render['decoder_hidden_dim'])
    else:
        raise ValueError('Rendering type ' + CFG_render['type'] + ' is not known!')

    return decoder
