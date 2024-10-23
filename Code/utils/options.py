import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--epoch',       type=int,   default=150,   help='epoch number') # 300
parser.add_argument('--lr',          type=float, default=5e-5,  help='learning rate')
parser.add_argument('--batchsize',   type=int,   default=10,    help='training batch size')
parser.add_argument('--trainsize',   type=int,   default=224,   help='training dataset size') #256
parser.add_argument('--clip',        type=float, default=0.5,   help='gradient clipping margin')
parser.add_argument('--lw',          type=float, default=0.001, help='weight')
parser.add_argument('--decay_rate',  type=float, default=0.5,   help='decay rate of learning rate') # 0.1
parser.add_argument('--decay_epoch', type=int,   default=200,    help='every n epochs decay learning rate') # 200
parser.add_argument('--load',        type=str,   default='./pretrained_model/swin_base_patch4_window7_224_22k.pth',  help='train from checkpoints')
parser.add_argument('--gpu_id',      type=str,   default='0',   help='train use gpu')


parser.add_argument('--rgb_label_root',      type=str, default='./train_2185/train_ori/train_images/',           help='the training rgb images root')
parser.add_argument('--depth_label_root',    type=str, default='./train_2185/train_ori/train_depth/',         help='the training depth images root')
parser.add_argument('--gt_label_root',       type=str, default='./train_2185/train_ori/train_masks/',            help='the training gt images root')
#
parser.add_argument('--val_rgb_root',        type=str, default='./train_2185/train_ori/train_images/',      help='the test rgb images root')
parser.add_argument('--val_depth_root',      type=str, default='./train_2185/train_ori/train_depth/',    help='the test depth images root')
parser.add_argument('--val_gt_root',         type=str, default='./train_2185/train_ori/train_masks/',       help='the test gt images root')

parser.add_argument('--save_path',           type=str, default='./Checkpoint/',    help='the path to save models and logs')


opt = parser.parse_args()

