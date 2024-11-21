from torch.nn import functional as F
import torch
import torch.nn as nn

from nets.vision_transformer import Block
from nets.backbone import Backbone, C2f, Conv
from nets.yolo_training import weights_init
from utils.utils_bbox import make_anchors


def fuse_conv_and_bn(conv, bn):
    # 混合Conv2d + BatchNorm2d 减少计算量
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          dilation=conv.dilation,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # 准备kernel
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # 准备bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


class DFL(nn.Module):
    # DFL模块
    # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        # bs, self.reg_max * 4, 8400
        b, c, a = x.shape
        # bs, 4, self.reg_max, 8400 => bs, self.reg_max, 4, 8400 => b, 4, 8400
        # 以softmax的方式，对0~16的数字计算百分比，获得最终数字。
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


# ---------------------------------------------------#
#   yolo_body
# ---------------------------------------------------#
class FENetBody(nn.Module):
    def __init__(self, input_shape, num_classes, phi='s', pretrained=False):
        super(FENetBody, self).__init__()
        depth_dict = {'n': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.00, }
        width_dict = {'n': 0.25, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }
        deep_width_dict = {'n': 1.00, 's': 1.00, 'm': 0.75, 'l': 0.50, 'x': 0.50, }
        dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3
        # -----------------------------------------------#
        #   输入图片是3, 640, 640
        # -----------------------------------------------#

        # ---------------------------------------------------#
        #   生成主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   256, 80, 80
        #   512, 40, 40
        #   1024 * deep_mul, 20, 20
        # ---------------------------------------------------#
        self.backbone = Backbone(base_channels, base_depth, deep_mul, phi, pretrained=pretrained)
        # ------------------------加强特征提取网络------------------------#
        self.frm1 = FRM(base_channels * 2, base_channels * 4)
        self.frm2 = FRM(base_channels * 4, base_channels * 8)
        self.frm3 = FRM(base_channels * 8, int(base_channels * 16 * deep_mul))
        self.gem = GEM(int(base_channels * 16 * deep_mul) + base_channels * 8, int(base_channels * 16 * deep_mul))
        self.ctm2 = AFM(int(base_channels * 16 * deep_mul) + base_channels * 8, base_channels * 8,
                        base_depth)
        self.ctm1 = AFM(base_channels * 8 + base_channels * 4, base_channels * 4,
                        base_depth)
        # ------------------------加强特征提取网络------------------------#
        ch = [base_channels * 4, base_channels * 8, int(base_channels * 16 * deep_mul)]
        self.shape = None
        self.nl = len(ch)
        # self.stride     = torch.zeros(self.nl)
        self.stride = torch.tensor(
            [256 / x.shape[-2] for x in self.backbone.forward(torch.zeros(1, 3, 256, 256))[1:]])  # forward
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = num_classes + self.reg_max * 4  # number of outputs per anchor
        self.num_classes = num_classes

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], num_classes)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, num_classes, 1)) for x in ch)
        if not pretrained:
            weights_init(self)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        return self

    def forward(self, x):
        #  backbone
        feat0, feat1, feat2, feat3 = self.backbone.forward(x)
        feat1 = self.frm1(feat0, feat1)
        feat2 = self.frm2(feat1, feat2)
        feat3 = self.frm3(feat2, feat3)
        # ------------------------加强特征提取网络------------------------#
        P5 = self.gem(feat3, feat2)
        P4 = self.ctm2(feat2, P5)
        P3 = self.ctm1(feat1, P4)
        # ------------------------加强特征提取网络------------------------#
        # P3 256, 80, 80
        # P4 512, 40, 40
        # P5 1024 * deep_mul, 20, 20
        shape = P3.shape  # BCHW

        # P3 256, 80, 80 => num_classes + self.reg_max * 4, 80, 80
        # P4 512, 40, 40 => num_classes + self.reg_max * 4, 40, 40
        # P5 1024 * deep_mul, 20, 20 => num_classes + self.reg_max * 4, 20, 20
        x = [P3, P4, P5]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        # num_classes + self.reg_max * 4 , 8400 =>  cls num_classes, 8400; 
        #                                           box self.reg_max * 4, 8400
        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split(
            (self.reg_max * 4, self.num_classes), 1)
        # origin_cls      = [xi.split((self.reg_max * 4, self.num_classes), 1)[1] for xi in x]
        dbox = self.dfl(box)
        return dbox, cls, x, self.anchors.to(dbox.device), self.strides.to(dbox.device)


class GEM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GEM, self).__init__()
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reduce = Conv(in_channels, out_channels, 1)
        self.trans = Block(out_channels, 4)

    def forward(self, x1, x2):
        x2 = self.downsample(x2)
        p = torch.cat([x1, x2], 1)
        c = self.reduce(p)
        out = self.trans(c) + c
        return out


class FRM(nn.Module):
    def __init__(self, channel1, channel2):
        super(FRM, self).__init__()
        self.conv = Conv(channel1, channel1, 3, 1)
        self.downSample = nn.Sequential(
            Conv(channel1, channel2, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.seg = nn.Sequential(
            nn.Conv2d(channel1, 1, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Sigmoid(),
        )
        self.attention = CBAM(channel2)

    def forward(self, x0, x1):
        x0 = self.conv(x0)
        seg = self.seg(x0)
        x0_down = self.downSample(x0)
        y = self.attention(x1 * seg + x0_down)
        return y


class AFM(nn.Module):
    def __init__(self, in_channels,  out_channels, base_depth):
        super(AFM, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.reduce = Conv(in_channels, out_channels, 1)
        self.c2f = C2f(out_channels, out_channels, base_depth, shortcut=False)
        self.trans = Block(out_channels, 4)
        self.weights = nn.Conv2d(out_channels * 2, 2, 1)
        self.fine = Conv(out_channels * 2, out_channels, 1)

    def forward(self, p1, p2):
        p2_upsample = self.upsample(p2)
        p = torch.cat([p2_upsample, p1], dim=1)
        p = self.reduce(p)
        c = self.c2f(p)
        t = self.trans(p)
        y = torch.cat([c, t], dim=1)
        weights = F.softmax(self.weights(y), dim=1)
        w1, w2  = torch.split(weights, [1, 1], dim=1)
        c = w1 * c
        t = w2 * t
        out = torch.cat([c, t], dim=1)
        out = self.fine(out) + p
        return out


class ChannelAttentionModule(nn.Module):
    def __init__(self, c1, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = c1 // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=c1, out_features=mid_channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_features=mid_channel, out_features=c1)
        )
        self.act = nn.Sigmoid()
        # self.act=nn.SiLU()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        return self.act(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.act = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.act(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, c1):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


if __name__ == '__main__':
    model =FENetBody((640, 640), 10, 's', False)
    x = torch.randn(size=(1, 3, 640, 640))
    # print(x.shape)
    y = model(x)

    print(y[-2].shape)
