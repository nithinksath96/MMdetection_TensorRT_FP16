# remove the last fc layer from checkpoints so that they can be finetuned on a new dataset

import argparse
from os.path import splitext

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--checkpoint', type=str, required=True)
    # parser.add_argument('--checkpoint', default='H:/Data/ModelZoo/mmdet/mask_rcnn_r50_fpn_2x_20181010-41d35c05.pth')
    parser.add_argument('--checkpoint', default='/data2/mengtial/ModelZoo/mmdet/mask_rcnn_r101_fpn_2x_20181129-a254bdfc.pth')
    opts = parser.parse_args()
    return opts

def main():
    opts = parse_args()
    ckpt = torch.load(opts.checkpoint)
    del ckpt['state_dict']['bbox_head.fc_cls.weight']
    del ckpt['state_dict']['bbox_head.fc_cls.bias']
    del ckpt['state_dict']['bbox_head.fc_reg.weight']
    del ckpt['state_dict']['bbox_head.fc_reg.bias']
    path, ext = splitext(opts.checkpoint)
    torch.save(ckpt, path + '_ft' + ext)

if __name__ == '__main__':
    main()
