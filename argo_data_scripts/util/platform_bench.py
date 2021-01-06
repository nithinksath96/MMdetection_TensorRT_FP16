import argparse, csv
from glob import glob
from os.path import join, isfile
from time import perf_counter

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2
from det import imread
from det.det_apis import init_detector, inference_detector

def parse_args():
    parser = argparse.ArgumentParser()
    # MT1080 Win10
    # parser.add_argument('--det-img-dir', type=str, default='D:/Data/ArgoVerse1.1/tracking/val/c9d6ebeb-be15-3df8-b6f1-5575bea8e6b9/ring_front_center')
    # parser.add_argument('--cifar-dir', type=str, default='D:/Data/SmallDB/CIFAR-10')
    # parser.add_argument('--iter', type=int, default=100)
    # parser.add_argument('--config', type=str, default='../mmdetection/configs/mask_rcnn_r50_fpn_1x.py')
    # parser.add_argument('--weights', type=str, default='D:/Data/ModelZoo/mmdet/mask_rcnn_r50_fpn_2x_20181010-41d35c05.pth')
    # parser.add_argument('--cpu-pre', action='store_true', default=False)
    # parser.add_argument('--out-dir', type=str, default='.')
    # parser.add_argument('--overwrite', action='store_true', default=False)

    # Trinity
    parser.add_argument('--det-img-dir', type=str, default='/data2/mengtial/ArgoVerse1.1/tracking/val/c9d6ebeb-be15-3df8-b6f1-5575bea8e6b9/ring_front_center')
    parser.add_argument('--cifar-dir', type=str, default='/data/mengtial/SmallDB/CIFAR-10')
    parser.add_argument('--iter', type=int, default=100)
    parser.add_argument('--config', type=str, default='~/repo/mmdetection/configs/mask_rcnn_r50_fpn_1x.py')
    parser.add_argument('--weights', type=str, default='/data2/mengtial/ModelZoo/mmdet/mask_rcnn_r50_fpn_2x_20181010-41d35c05.pth')
    parser.add_argument('--cpu-pre', action='store_true', default=False)
    parser.add_argument('--out-dir', type=str, default='.')
    parser.add_argument('--overwrite', action='store_true', default=False)

    # MTRTX
    # parser.add_argument('--det-img-dir', type=str, default='H:/Data/ArgoVerse1.1/tracking/val/c9d6ebeb-be15-3df8-b6f1-5575bea8e6b9/ring_front_center')
    # parser.add_argument('--cifar-dir', type=str, default='H:/Data/SmallDB/CIFAR-10')
    # parser.add_argument('--iter', type=int, default=100)
    # parser.add_argument('--config', type=str, default='../mmdetection/configs/mask_rcnn_r50_fpn_1x.py')
    # parser.add_argument('--weights', type=str, default='H:/Data/ModelZoo/mmdet/mask_rcnn_r50_fpn_2x_20181010-41d35c05.pth')
    # parser.add_argument('--cpu-pre', action='store_true', default=False)
    # parser.add_argument('--out-dir', type=str, default='.')
    # parser.add_argument('--overwrite', action='store_true', default=False)

    # parser.add_argument('--det-img-dir', type=str, default='D:/Data/PlatformBench/Det')
    # parser.add_argument('--cifar-dir', type=str, default='D:/Data/PlatformBench/CIFAR-10')
    # parser.add_argument('--iter', type=int, default=100)
    # parser.add_argument('--config', type=str, default='../mmdetection/configs/mask_rcnn_r50_fpn_1x.py')
    # parser.add_argument('--weights', type=str, default='D:/Data/ModelZoo/mmdet/mask_rcnn_r50_fpn_2x_20181010-41d35c05.pth')
    # parser.add_argument('--cpu-pre', action='store_true', default=False)
    # parser.add_argument('--out-dir', type=str, default='.')
    # parser.add_argument('--overwrite', action='store_true', default=False)
    opts = parser.parse_args()
    return opts

def det_test(opts, scale, imgs):
    assert len(imgs)
    opts.in_scale = scale
    opts.no_mask = True
    model = init_detector(opts)

    # warm up the GPU
    _ = inference_detector(model, imgs[0])
    torch.cuda.synchronize()

    runtime = []
    for img in imgs:
        t1 = perf_counter()
        _ = inference_detector(model, img, gpu_pre=not opts.cpu_pre)
        torch.cuda.synchronize()
        t2 = perf_counter()
        runtime.append(t2 - t1)

    return sum(runtime)/len(runtime)

def classify_test(opts):
    # adapted from mtpt
    import torch.nn as nn
    import torch.nn.functional as F

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1):
            super(BasicBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out

    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, in_planes, planes, stride=1):
            super(Bottleneck, self).__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(self.expansion*planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out

    class ResNet(nn.Module):
        def __init__(self, block, num_blocks, num_classes=10):
            super(ResNet, self).__init__()
            self.in_planes = 64

            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.linear = nn.Linear(512*block.expansion, num_classes)

        def _make_layer(self, block, planes, num_blocks, stride):
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out

    def ResNet18(num_classes=10):
        return ResNet(BasicBlock, [2,2,2,2], num_classes)

    def ResNet34(num_classes=10):
        return ResNet(BasicBlock, [3,4,6,3], num_classes)

    def ResNet50(num_classes=10):
        return ResNet(Bottleneck, [3,4,6,3], num_classes)

    def ResNet101(num_classes=10):
        return ResNet(Bottleneck, [3,4,23,3], num_classes)

    def ResNet152(num_classes=10):
        return ResNet(Bottleneck, [3,8,36,3], num_classes)


    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_set = torchvision.datasets.CIFAR10(root=opts.cifar_dir, train=False,
                                        download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(test_set, batch_size=128,
                                            shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128,
                                            shuffle=False, num_workers=4)
    net = ResNet18().to('cuda')
    net.eval()

    runtime = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if i >= opts.iter + 5:
                break
            x = data[0].to('cuda')
            torch.cuda.synchronize()
            t1 = perf_counter()
            _ = net(x)
            torch.cuda.synchronize()
            t2 = perf_counter()
            if i < 5:
                continue
            runtime.append(t2 - t1)
    inference_time = sum(runtime)/len(runtime)

    import torch.optim as optim
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

    runtime = []
    for i, data in enumerate(train_loader):
        if i >= opts.iter + 5:
            break
        x, y = data
        x, y = x.to('cuda'), y.to('cuda')
        torch.cuda.synchronize()
        t1 = perf_counter()

        y_out = net(x)
        loss = criterion(y_out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        t2 = perf_counter()
        if i < 5:
            continue
        runtime.append(t2 - t1)
    train_time = sum(runtime)/len(runtime)
    return inference_time, train_time

def main():
    opts = parse_args()

    mkdir2(opts.out_dir)

    print('Reading detection images')
    det_img_files = sorted(glob(join(opts.det_img_dir, '*.jpg')))
    det_imgs = []
    for path in det_img_files:
        det_imgs.append(imread(path))

    print('Detection Test at Scale 0.5')
    rt_det1 = det_test(opts, 0.5, det_imgs)
    print(f'Mean runtime (ms): {1e3*rt_det1:.3g}')

    print('Detection Test at Scale 1')
    rt_det2 = det_test(opts, 1, det_imgs)
    print(f'Mean runtime (ms): {1e3*rt_det2:.3g}')
    det_imgs = None

    print('Classification Test')
    rt_inf, rt_train = classify_test(opts)
    print(f'Inference Mean runtime (ms): {1e3*rt_inf:.3g}')
    print(f'Training Mean runtime (ms): {1e3*rt_train:.3g}')

    out_path = join(opts.out_dir, 'platform_bench.txt')
    if opts.overwrite or not isfile(out_path):
        with open(out_path, 'w', newline='\n') as f:
            writer = csv.writer(f)
            writer.writerow([rt_det1, rt_det2, rt_inf, rt_train])
        print(f'Saved results to "{out_path}"')

if __name__ == '__main__':
    main()