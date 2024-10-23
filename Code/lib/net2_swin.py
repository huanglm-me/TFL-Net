import torch
from torch import nn
import torch.nn.functional as F
from models.SwinT import SwinTransformer

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class TGL_Fusion(nn.Module):
    def __init__(self, in_channel, out_channel):
            super(TGL_Fusion, self).__init__()

            self.support = nn.Conv2d(in_channel, in_channel, 1, 1)

            self.query = nn.Conv2d(in_channel, in_channel, 1, 1)

            self.g_xf = nn.Sequential(
                nn.Conv2d(in_channel, in_channel, 1, 1),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(inplace=True),
            )

            self.g_zf = nn.Sequential(
                nn.Conv2d(in_channel, in_channel, 1, 1),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(inplace=True),
            )

            self.fi = nn.Sequential(
                nn.Conv2d(in_channel * 2, out_channel, 1, 1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
            )

    def forward(self, zf, xf):
            xf_trans = self.query(xf)
            zf_trans = self.support(zf)

            xf_g = self.g_xf(xf)
            zf_g = self.g_zf(zf)

            # calculate similarity
            shape_x = xf_trans.shape
            shape_z = zf_trans.shape

            zf_trans_plain = zf_trans.view(-1, shape_z[1], shape_z[2] * shape_z[3])
            zf_g_plain = zf_g.view(-1, shape_z[1], shape_z[2] * shape_z[3]).permute(0, 2, 1)
            xf_trans_plain = xf_trans.view(-1, shape_x[1], shape_x[2] * shape_x[3]).permute(0, 2, 1)

            similar1 = torch.matmul(xf_trans_plain, zf_trans_plain)
            similar2 = torch.matmul(zf_trans_plain.permute(0, 2, 1), zf_trans_plain)
            similar = similar1 - 0.1*similar2

            similar = F.softmax(similar, dim=2)  # normalize(similar)
            embedding = torch.matmul(similar, zf_g_plain).permute(0, 2, 1)
            embedding = embedding.view(-1, shape_x[1], shape_x[2], shape_x[3])

            # aggregated feature
            output = torch.cat([embedding, xf_g], 1)
            output = self.fi(output)
            return output

class TGLNet(nn.Module):
    def __init__(self):
        super(TGLNet,self,).__init__()
        self.rgb_swin = SwinTransformer(img_size=224, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=7)
        self.depth_swin = SwinTransformer(img_size=224, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=7)

        self.linear_out4 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear_out3 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear_out2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear_out1 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)

        self.tgl4_rgb = TGL_Fusion(256, 256)
        self.tgl3_rgb = TGL_Fusion(256, 256)
        self.tgl2_rgb = TGL_Fusion(256, 256)
        self.tgl1_rgb = TGL_Fusion(256, 256)

        self.tgl4_h = TGL_Fusion(256, 256)
        self.tgl3_h = TGL_Fusion(256, 256)
        self.tgl2_h = TGL_Fusion(256, 256)
        self.tgl1_h = TGL_Fusion(256, 256)

        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Sequential(BasicConv2d(256 * 2, 256, 3, padding=1), self.relu)
        self.conv3 = nn.Sequential(BasicConv2d(256 * 3, 256, 3, padding=1), self.relu)
        self.conv2 = nn.Sequential(BasicConv2d(256 * 3, 256, 3, padding=1), self.relu)
        self.conv1 = nn.Sequential(BasicConv2d(256 * 3, 256, 3, padding=1), self.relu)
        self.conv0 = nn.Sequential(BasicConv2d(256 * 2, 256, 3, padding=1), self.relu)

        self.upsample_4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self,rgb,h):
        xsize = rgb.size()[2:]

        rgb1, rgb2, rgb3, rgb4 = self.rgb_swin(rgb)
        h1, h2, h3, h4 = self.depth_swin(h)

        # rgb
        rgb4_4 = self.tgl4_rgb(rgb4, rgb4)
        rgb44 = torch.cat((rgb4, rgb4_4), 1)
        rgb44 = self.conv4(rgb44)
        rgb44 = self.upsample_4(rgb44)

        rgb3_4 = self.tgl3_rgb(rgb44, rgb3)
        rgb4_3 = self.tgl3_rgb(rgb3, rgb44)
        rgb33 = torch.cat((rgb44, rgb3_4, rgb4_3), 1)
        rgb33 = self.conv3(rgb33)
        rgb33 = self.upsample_3(rgb33)

        rgb2_3 = self.tgl2_rgb(rgb33, rgb2)
        rgb3_2 = self.tgl2_rgb(rgb2, rgb33)
        rgb22 = torch.cat((rgb33, rgb2_3, rgb3_2), 1)
        rgb22 = self.conv2(rgb22)
        rgb22 = self.upsample_2(rgb22)

        rgb1_2 = self.tgl1_rgb(rgb22, rgb1)
        rgb2_1 = self.tgl1_rgb(rgb1, rgb22)
        rgb11 = torch.cat((rgb22, rgb1_2, rgb2_1), 1)
        rgb11 = self.conv1(rgb11)

        # t/d
        h4_4 = self.tgl4_h(h4, h4)
        h44 = torch.cat((h4, h4_4), 1)
        h44 = self.conv4(h44)
        h44 = self.upsample_4(h44)

        h3_4 = self.tgl3_h(h44, h3)
        h4_3 = self.tgl3_h(h3, h44)
        h33 = torch.cat((h44, h3_4, h4_3), 1)
        h33 = self.conv3(h33)
        h33 = self.upsample_3(h33)

        h2_3 = self.tgl2_h(h33, h2)
        h3_2 = self.tgl2_h(h2, h33)
        h22 = torch.cat((h33, h2_3, h3_2), 1)
        h22 = self.conv2(h22)
        h22 = self.upsample_2(h22)

        h1_2 = self.tgl1_h(h22, h1)
        h2_1 = self.tgl1_h(h1, h22)
        h11 = torch.cat((h22, h1_2, h2_1), 1)
        h11 = self.conv1(h11)

        rgbh11 = torch.cat((rgb11, h11), 1)
        rgbh1 = self.conv0(rgbh11)

        score1 = F.interpolate(self.linear_out4(rgb11), size=xsize, mode='bilinear', align_corners=True)
        score2 = F.interpolate(self.linear_out3(h11), size=xsize, mode='bilinear', align_corners=True)
        score3 = F.interpolate(self.linear_out2(rgb44), size=xsize, mode='bilinear', align_corners=True)
        score4 = F.interpolate(self.linear_out2(h44), size=xsize, mode='bilinear', align_corners=True)
        score = F.interpolate(self.linear_out1(rgbh1), size=xsize, mode='bilinear', align_corners=True)

        return score,score1,score2,score3, score4

    def load_pre(self, pre_model):
        self.rgb_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model}")
        self.depth_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"Depth SwinTransformer loading pre_model ${pre_model}")

