import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from gcn_lib import Grapher, act_layer
import math



class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x  # .reshape(B, C, N, 1)


class Stem(nn.Module):  # 采用Overlap进行切块
    """ Image to Visual Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """

    def __init__(self, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim // 2),
            act_layer(act),
            nn.Conv2d(out_dim // 2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        print(x.shape)
        return x


class Downsample(nn.Module):
    """ Convolution-based downsample
    """

    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class OverlapPatchMerge(nn.Module):
    # OverlapPatchMerge is used
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)

        return x, H, W


# DecodeBlock
class Skip_connect(nn.Module):
    def __init__(self):
        super(Skip_connect, self).__init__()

    def forward(self, x1, x2):
        out = torch.cat((x1, x2), dim=1)
        return out


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DecoderHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, num_classes=2, in_channels=None, embedding_dim=768, dropout_ratio=0.1):
        super(DecoderHead, self).__init__()
        if in_channels is None:
            in_channels = [128, 256, 512, 1024]
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        ############### 统一通道数 #################
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim * 4,
            c2=embedding_dim,
            k=1,
        )

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=True)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=True)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=True)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))  # 特征融合

        return _c


class Channel_attention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(Channel_attention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # .view是进行reshape
        max_pool_out = self.max_pool(x).view([b, c])
        avg_pool_out = self.avg_pool(x).view([b, c])

        max_fc_out = self.fc(max_pool_out)
        avg_fc_out = self.fc(avg_pool_out)

        out = max_fc_out + avg_fc_out
        out = self.sigmoid(out).view([b, c, 1, 1])

        return out * x


class Weighted_MultiScale_Feature_Module(nn.Module):
    def __init__(self, in_channels_list, out_channels, ppm_scales=(1, 2, 3, 6), embedding_dim=512):
        super(Weighted_MultiScale_Feature_Module, self).__init__()

        # PPM分支
        self.ppm_branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveMaxPool2d(output_size=(scale, scale)),
                nn.Conv2d(in_channels_list[-1], out_channels, kernel_size=1)
            ) for scale in ppm_scales
        ])

        self.linear_fuse = nn.Conv2d(embedding_dim * len(ppm_scales), embedding_dim, kernel_size=1)

        # 全局平均池化分支
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_adjustment = nn.Sequential(
            nn.Conv2d(in_channels_list[-1], out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        x1, x2, x3, x4 = inputs

        # PPM分支
        ppm_results = [branch(x4) for branch in self.ppm_branches]
        ppm_results = [F.interpolate(result, size=(80, 80), mode='bilinear', align_corners=True) for result in ppm_results]

        # 融合四个尺度特征图
        ppm_results1 = self.linear_fuse(torch.cat(ppm_results, dim=1))

        # 全局平均池化分支
        global_avg_pool = self.global_avg_pool(x4)
        channel_weights = self.channel_adjustment(global_avg_pool)

        # 对PPM分支结果加权
        output = ppm_results1 * channel_weights
        return output


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=4, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        # self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Feature_Fusion_Module(nn.Module):
    ############## 深度加权特征融合模块 ##################
    def __init__(self, num_classes=1, dropout_ratio=.1, embedding_dim=1024):
        super(Feature_Fusion_Module, self).__init__()

        self.in_channels1 = 1024
        self.in_channels2 = 512
        self.out_channels = 1024

        # 连接前两个模块输出的特征图
        self.conbnrelu = ConvBNReLU(self.in_channels1 + self.in_channels2, self.out_channels, ks=1, stride=1, padding=0)

        # 分支2：全局平均池化 + 全连接 + Sigmoid
        self.global_avg_pool = nn.AdaptiveAvgPool2d(7)  # 变成7x7的特征图
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, x1, x2):
        # 连接前两个模块输出的特征图
        x = torch.cat([x1, x2], dim=1)
        x = self.conbnrelu(x)  # 生成连接后的特征图

        # 分支2：全局平均池化 + 全连接 + Sigmoid  构建权重
        global_avg = self.global_avg_pool(x)
        weights = self.sigmoid(self.conv2(global_avg))
        weights = F.interpolate(weights, size=(x.shape[-1], x.shape[-1]), mode='nearest')
        # 对深度混合特征图进行重新加权
        weights_x = x * weights

        # 最终的模块输出：原始深度混合特征图 + 重新加权后的深度混合特征图
        output = weights_x + x

        final_x = self.dropout(output)
        last_x = self.linear_pred(final_x)

        return last_x


class DeepGCN(torch.nn.Module):
    def __init__(self, num_k=9, act='gelu', norm='batch', bias=True, epsilon=0.2, use_stochastic=False, conv='mr',
                 emb_dims=1024, blocks=None, drop_path=0., channels=None, img_height: int = 320, img_width: int = 320):
        super(DeepGCN, self).__init__()
        if channels is None:
            channels = [128, 256, 512, 1024]
        if blocks is None:
            blocks = [2, 2, 18, 2]
        stochastic = use_stochastic  # 优化GCN，作用类似于dropout
        k = num_k
        self.height = img_height
        self.width = img_width
        self.blocks = blocks
        self.n_blocks = sum(blocks)
        self.stem = Stem(out_dim=channels[0], act=act)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], self.height // 4, self.width // 4))  # 位置编码，按224的大小计算的，若要改输入大小，这里也要调整
        channels = channels
        emb_dims = emb_dims
        reduce_ratios = [4, 2, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]  # number of knn's k
        max_dilation = 49 // max(num_knn)  # 用于构建图结构的膨胀系数，膨胀knn算法所需

        HW = self.height // 4 * self.width // 4

        ################################################################################################
        #  对于每个Vig block进行填充，其中包含下采样模块Downsample(上一层输出c，本层输出c)，Grapher，FNN
        #  整个这段是Encoder的关键部分，也就是总共的24个block,与Segformer不一样的是没采用overlap patch merging所以就少了几层的patch_embedding
        ###############################################################
        # stage1 2个block
        ###############################################################
        stage1 = []
        idx_1 = 0
        for b in range(blocks[0]):
            stage1.append(
                Seq(
                    Grapher(
                        channels[0], num_knn[idx_1], min(idx_1 // 4 + 1, max_dilation), conv, act, norm,
                        bias, stochastic, epsilon, reduce_ratios[0], n=HW, drop_path=dpr[idx_1],
                        relative_pos=True
                    ),
                    FFN(
                        channels[0], channels[0] * 4, act=act, drop_path=dpr[idx_1]
                    )
                )
            )
            idx_1 += 1
        self.stage1 = nn.ModuleList(stage1)
        # print("Stage1:", self.stage1)
        # print("================================================")

        ################################################################
        # stage2 2个block
        ################################################################

        stage2 = []
        # idx_1 = 0
        stage2.append(Downsample(channels[0], channels[1])) # 做1x1卷积，用于扩充通道数
        for i in range(blocks[1]):
            stage2.append(
                Seq(
                    Grapher(
                        channels[1], num_knn[idx_1], min(idx_1 // 4 + 1, max_dilation), conv, act, norm,
                        bias, stochastic, epsilon, reduce_ratios[1], n=HW, drop_path=dpr[idx_1],
                        relative_pos=True
                    ),
                    FFN(
                        channels[1], channels[1] * 4, act=act, drop_path=dpr[idx_1]
                    )
                )
            )
            idx_1 += 1
        HW = HW // 4  # 调整分辨率
        self.stage2 = nn.ModuleList(stage2)
        self.patch_merged2 = OverlapPatchMerge(patch_size=3, stride=2, in_chans=128, embed_dim=128)
        # print("Stage2:", self.stage2)
        # print("================================================")

        ################################################################
        # stage3 18个block
        ################################################################

        stage3 = []
        # idx_1 = 0

        stage3.append(Downsample(channels[1], channels[2]))
        for i in range(blocks[2]):
            stage3.append(
                Seq(
                    Grapher(
                        channels[2], num_knn[idx_1], min(idx_1 // 4 + 1, max_dilation), conv, act, norm,
                        bias, stochastic, epsilon, reduce_ratios[2], n=HW, drop_path=dpr[idx_1],
                        relative_pos=True
                    ),
                    FFN(
                        channels[2], channels[2] * 4, act=act, drop_path=dpr[idx_1]
                    )
                )
            )
            idx_1 += 1
        HW = HW // 4
        self.stage3 = nn.ModuleList(stage3)
        self.patch_merged3 = OverlapPatchMerge(patch_size=3, stride=2, in_chans=256, embed_dim=256)
        ################################################################
        # stage4 2个block
        ################################################################

        stage4 = []

        stage4.append(Downsample(channels[2], channels[3]))
        for i in range(blocks[3]):
            stage4.append(
                Seq(
                    Grapher(
                        channels[3], num_knn[idx_1], min(idx_1 // 4 + 1, max_dilation), conv, act, norm,
                        bias, stochastic, epsilon, reduce_ratios[3], n=HW, drop_path=dpr[idx_1],
                        relative_pos=True
                    ),
                    FFN(
                        channels[3], channels[3] * 4, act=act, drop_path=dpr[idx_1]
                    )
                )
            )
            idx_1 += 1
        HW = HW // 4
        self.stage4 = nn.ModuleList(stage4)
        self.patch_merged4 = OverlapPatchMerge(patch_size=3, stride=2, in_chans=512, embed_dim=512)
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        x = self.stem(inputs) + self.pos_embed
        block_outs = []
        # stage1
        for i, blk in enumerate(self.stage1):
            x = blk.forward(x)
        block_outs.append(x)

        # stage2
        x, H, W = self.patch_merged2.forward(x)
        for i, blk in enumerate(self.stage2):
            x = blk.forward(x)
        block_outs.append(x)

        # stage3
        x, H, W = self.patch_merged3.forward(x)
        for i, blk in enumerate(self.stage3):
            x = blk.forward(x)
        block_outs.append(x)

        # stage4
        x, H, W = self.patch_merged4.forward(x)
        for i, blk in enumerate(self.stage4):
            x = blk.forward(x)
        block_outs.append(x)
        return x, block_outs


class GraphCrack(nn.Module):
    def __init__(self, num_classes=1, channels=None, emb_dimes=1024):
        super(GraphCrack, self).__init__()
        if channels is None:
            channels = [128, 256, 512, 1024]
        self.backbone = DeepGCN()
        self.decode_head1 = DecoderHead(num_classes, channels, emb_dimes)
        self.decode_head2 = Weighted_MultiScale_Feature_Module(channels, 512, embedding_dim=512)
        self.Final_output = Feature_Fusion_Module(num_classes)

    def forward(self, inputs):
        x, block_outs = self.backbone.forward(inputs)
        H, W = inputs.size(2), inputs.size(3)
        o1 = self.decode_head1.forward(block_outs)
        o2 = self.decode_head2.forward(block_outs)
        o3 = self.Final_output(o1, o2)
        o4 = F.interpolate(o3, scale_factor=4, mode='bilinear', align_corners=True)  # 最终的上采样结果.4倍上采样还原到原图
        return o4
