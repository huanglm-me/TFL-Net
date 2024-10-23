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

        self.upsample_4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self,rgb,h):
        xsize = rgb.size()[2:]

        rgb1, rgb2, rgb3, rgb4 = self.rgb_swin(rgb)
        h1, h2, h3, h4 = self.depth_swin(h)

        rgb4 = self.tgl4_rgb(h4, rgb4)
        h4 = self.tgl4_h(rgb4, h4)
        rgbh4 = torch.cat((rgb4, h4), 1)
        rgbh4 = self.conv4(rgbh4)
        rgbh4 = self.upsample_4(rgbh4)

        rgb3 = self.tgl3_rgb(rgbh4, rgb3)
        h3 = self.tgl3_h(rgbh4, h3)
        rgbh3 = torch.cat((rgbh4, rgb3, h3), 1)
        rgbh3 = self.conv3(rgbh3)
        rgbh3 = self.upsample_3(rgbh3)

        rgb2 = self.tgl2_rgb(rgbh3, rgb2)
        h2 = self.tgl2_h(rgbh3, h2)
        rgbh2 = torch.cat((rgbh3, rgb2, h2), 1)
        rgbh2 = self.conv2(rgbh2)
        rgbh2 = self.upsample_2(rgbh2)

        rgb1 = self.tgl1_rgb(rgbh2, rgb1)
        h1 = self.tgl1_h(rgbh2, h1)
        rgbh1 = torch.cat((rgbh2, rgb1, h1), 1)
        rgbh1 = self.conv1(rgbh1)

        score1 = F.interpolate(self.linear_out4(rgbh4), size=xsize, mode='bilinear', align_corners=True)
        score2 = F.interpolate(self.linear_out3(rgbh3), size=xsize, mode='bilinear', align_corners=True)
        score3 = F.interpolate(self.linear_out2(rgbh2), size=xsize, mode='bilinear', align_corners=True)
        score = F.interpolate(self.linear_out1(rgbh1), size=xsize, mode='bilinear', align_corners=True)

        return score,score1,score2,score3

    def load_pre(self, pre_model):
        self.rgb_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model}")
        self.depth_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"Depth SwinTransformer loading pre_model ${pre_model}")

