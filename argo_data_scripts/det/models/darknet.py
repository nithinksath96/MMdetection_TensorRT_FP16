from os.path import basename
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.registry import BACKBONES

class weightedFeatureFusion(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers, weight=False):
        super(weightedFeatureFusion, self).__init__()
        self.layers = layers  # layer indices
        self.weight = weight  # apply weights boolean
        self.n = len(layers) + 1  # number of layers
        if weight:
            self.w = torch.nn.Parameter(torch.zeros(self.n))  # layer weights

    def forward(self, x, outputs):
        # Weights
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)  # sigmoid weights (0-1)
            x = x * w[0]

        # Fusion
        nc = x.shape[1]  # input channels
        for i in range(self.n - 1):
            a = outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[self.layers[i]]  # feature to add
            ac = a.shape[1]  # feature channels
            dc = nc - ac  # delta channels

            # Adjust channels
            if dc > 0:  # slice input
                x[:, :ac] = x[:, :ac] + a  # or a = nn.ZeroPad2d((0, 0, 0, 0, 0, dc))(a); x = x + a
            elif dc < 0:  # slice feature
                x = x + a[:, :nc]
            else:  # same shape
                x = x + a
        return x

@BACKBONES.register_module
class Darknet(nn.Module):
    """Darknet backbone.
    """

    def __init__(
        self,
        cfg=None,
        out_indices=(1, 2, 3),
    ):
        super().__init__()
        if cfg is None:
            from os.path import dirname, realpath, join
            filedir = dirname(realpath(__file__))
            cfg = join(filedir, 'darknet53.cfg')
        self.module_defs, self.stage_last_layer = self.parse_model_cfg(cfg)
        # note stage_last_layer is the indices in module_defs, not neccesarily pytorch layers
        self.out_indices = out_indices
        self.out_layers = [self.stage_last_layer[i] for i in out_indices]

        self.module_list, self.routs = self.create_modules(self.module_defs)

        # possible extension if training takes too long
        # self._freeze_stages()

    def parse_model_cfg(self, path):
        # Parse the yolo *.cfg file and return module definitions path may be 'cfg/yolov3.cfg', 'yolov3.cfg', or 'yolov3'
        if not path.endswith('.cfg'):  # add .cfg suffix if omitted
            path += '.cfg'

        with open(path, 'r') as f:
            lines = f.read().split('\n')
        lines = [x for x in lines if x and not x.startswith('#')]
        lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
        mdefs = []  # module definitions
        stage_last_layer = []
        for line in lines:
            if line.startswith('['):  # This marks the start of a new block
                mdefs.append({})
                mdefs[-1]['type'] = line[1:-1].rstrip()
                if mdefs[-1]['type'] == 'convolutional':
                    mdefs[-1]['batch_normalize'] = 0  # pre-populate with zeros (may be overwritten later)
            elif line.startswith('@stage'):
                stage_last_layer.append(len(mdefs) - 2) # there is a pop(0) in create_modules
            else:
                key, val = line.split("=")
                key = key.rstrip()

                if key == 'anchors':  # return nparray
                    mdefs[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))  # np anchors
                elif key in ['from', 'layers', 'mask']:  # return array
                    mdefs[-1][key] = [int(x) for x in val.split(',')]
                else:
                    val = val.strip()
                    if val.isnumeric():  # return int or float
                        mdefs[-1][key] = int(val) if (int(val) - float(val)) == 0 else float(val)
                    else:
                        mdefs[-1][key] = val  # return string

        # Check all fields are supported
        supported = ['type', 'batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation', 'layers', 'groups',
                    'from', 'mask', 'anchors', 'classes', 'num', 'jitter', 'ignore_thresh', 'truth_thresh', 'random',
                    'stride_x', 'stride_y', 'weights_type', 'weights_normalization', 'scale_x_y', 'beta_nms', 'nms_kind',
                    'iou_loss', 'iou_normalizer', 'cls_normalizer', 'iou_thresh']

        f = []  # fields
        for x in mdefs[1:]:
            [f.append(k) for k in x if k not in f]
        u = [x for x in f if x not in supported]  # unsupported fields
        assert not any(u), "Unsupported fields %s in %s. See https://github.com/ultralytics/yolov3/issues/631" % (u, path)

        return mdefs, stage_last_layer

    def create_modules(self, module_defs):
        # Constructs module list of layer blocks from module configuration in module_defs

        hyperparams = module_defs.pop(0)
        output_filters = [int(hyperparams['channels'])]
        module_list = nn.ModuleList()
        routs = []  # list of layers which rout to deeper layers

        for i, mdef in enumerate(module_defs):
            modules = nn.Sequential()

            if mdef['type'] == 'convolutional':
                bn = mdef['batch_normalize']
                filters = mdef['filters']
                size = mdef['size']
                stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
                modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                    out_channels=filters,
                                                    kernel_size=size,
                                                    stride=stride,
                                                    padding=(size - 1) // 2 if mdef['pad'] else 0,
                                                    groups=mdef['groups'] if 'groups' in mdef else 1,
                                                    bias=not bn))
                if bn:
                    modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))
                if mdef['activation'] == 'leaky':  # activation study https://github.com/ultralytics/yolov3/issues/441
                    modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))

            elif mdef['type'] == 'maxpool':
                size = mdef['size']
                stride = mdef['stride']
                maxpool = nn.MaxPool2d(kernel_size=size, stride=stride, padding=(size - 1) // 2)
                if size == 2 and stride == 1:  # yolov3-tiny
                    modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                    modules.add_module('MaxPool2d', maxpool)
                else:
                    modules = maxpool

            elif mdef['type'] == 'upsample':
                modules = nn.Upsample(scale_factor=mdef['stride'])

            elif mdef['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
                layers = mdef['layers']
                filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
                routs.extend([l if l > 0 else l + i for l in layers])

            elif mdef['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
                layers = mdef['from']
                filters = output_filters[-1]
                routs.extend([i + l if l < 0 else l for l in layers])
                modules = weightedFeatureFusion(layers=layers, weight='weights_type' in mdef)

            else:
                print('Warning: Unrecognized Layer Type: ' + mdef['type'])

            # Register module list and number of output filters
            module_list.append(modules)
            output_filters.append(filters)

        return module_list, routs

    def load_darknet_weights(self, path, cutoff=-1):
        # Parses and loads the weights stored in 'weights'

        # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
        
        filename = basename(path)
        if filename == 'darknet53.conv.74':
            cutoff = 75
        elif filename == 'yolov3-tiny.conv.15':
            cutoff = 15

        # Read weights file
        with open(path, 'rb') as f:
            # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
            version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
            seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training

            weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

        ptr = 0
        for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if mdef['type'] == 'convolutional':
                conv = module[0]
                if mdef['batch_normalize']:
                    # Load BN bias, weights, running mean and running variance
                    bn = module[1]
                    nb = bn.bias.numel()  # number of biases
                    # Bias
                    bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.bias))
                    ptr += nb
                    # Weight
                    bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.weight))
                    ptr += nb
                    # Running Mean
                    bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_mean))
                    ptr += nb
                    # Running Var
                    bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_var))
                    ptr += nb
                else:
                    # Load conv. bias
                    nb = conv.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + nb]).view_as(conv.bias)
                    conv.bias.data.copy_(conv_b)
                    ptr += nb
                # Load conv. weights
                nw = conv.weight.numel()  # number of weights
                conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nw]).view_as(conv.weight))
                ptr += nw

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            self.load_darknet_weights(pretrained)

    # def _freeze_stages(self):
    #     if self.frozen_stages >= 0:
    #         self.norm1.eval()
    #         for m in [self.conv1, self.norm1]:
    #             for param in m.parameters():
    #                 param.requires_grad = False

    #     for i in range(1, self.frozen_stages + 1):
    #         m = getattr(self, 'layer{}'.format(i))
    #         m.eval()
    #         for param in m.parameters():
    #             param.requires_grad = False

    def forward(self, x):
        outs = []
        outs_for_shortcut = []
        verbose = False
        if verbose:
            str = ''
            print('0', x.shape)

        for i, (mdef, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = mdef['type']
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif mtype == 'shortcut':  # sum
                if verbose:
                    l = [i - 1] + module.layers  # layers
                    s = [list(x.shape)] + [list(outs_for_shortcut[i].shape) for i in module.layers]  # shapes
                    str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, s)])
                x = module(x, outs_for_shortcut)  # weightedFeatureFusion()
            elif mtype == 'route':  # concat
                layers = mdef['layers']
                if verbose:
                    l = [i - 1] + layers  # layers
                    s = [list(x.shape)] + [list(outs_for_shortcut[i].shape) for i in layers]  # shapes
                    str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, s)])
                if len(layers) == 1:
                    x = outs_for_shortcut[layers[0]]
                else:
                    try:
                        x = torch.cat([out[i] for i in layers], 1)
                    except:  # apply stride 2 for darknet reorg layer
                        outs_for_shortcut[layers[1]] = F.interpolate(outs_for_shortcut[layers[1]], scale_factor=[0.5, 0.5])
                        x = torch.cat([outs_for_shortcut[i] for i in layers], 1)

            outs_for_shortcut.append(x if i in self.routs else [])
            if i in self.out_layers:
                outs.append(x)

            if verbose:
                print('%g/%g %s -' % (i, len(self.module_list), mtype), list(x.shape), str)
                str = ''

        return tuple(outs)

