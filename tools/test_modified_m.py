import argparse
import os
import os.path as osp
import pickle
import shutil
import tempfile

import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

from time import perf_counter

#import multiprocessing
#multiprocessing.set_start_method('spawn', True)
import pdb

#from mmdet.models.detectors import load_trt


def single_gpu_test(model, data_loader,show=False ):
    model.eval()
    results = []
    #pdb.set_trace()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    elapsed_time=0.0
    total_elapsed_time=0.0
    training_examples=0.0
    #pdb.set_trace()
    #load_trt.init()

    for i, data in enumerate(data_loader):
        #with torch.autograd.profiler.profile(use_cuda=True) as prof:
        with torch.no_grad():
            #sychronise torch. Executes the command in order
            #print("Data", len(data['img']))
            # b,c,h,w  = data['img'][0].shape
            # h1 = int((h*scale))
            # w1 = int((w*scale))
            # data['img'][0] = data['img'][0][:,:,:h1+1,:w1+1]
            t1_start = perf_counter()
            #pdb.set_trace()
            result = model(return_loss=False, rescale=not show, **data)
            
            torch.cuda.synchronize() 
            t1_stop = perf_counter()
            elapsed_time+=t1_stop-t1_start
            training_examples=training_examples+1

            if(i%100==0):
                print("Average Elapsed time",elapsed_time/100.0)
                total_elapsed_time+=elapsed_time
                elapsed_time=0.0
            #print("-----------------------------------------------------------------------")
            #pdb.set_trace()
            #print("Result",len(result))

        results.append(result)

        if show:
           
            model.module.show_result(data, result)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            
            prog_bar.update()
    #print(prof)
    from mmdet.models.detectors.single_stage import time_torch2trt
    init_time_single = time_torch2trt
    from mmdet.models.detectors.two_stage import time_torch2trt
    init_time_two = time_torch2trt
    print("\n Total ELAPSED TIME",total_elapsed_time)
    print("\n Training examples",training_examples)
    print("Time taken to initialize torch2trt",max(init_time_single,init_time_two))
    avg_elapsed_time = ((total_elapsed_time-max(init_time_single,init_time_two))/training_examples)
    #avg_elapsed_time = ((total_elapsed_time)/training_examples)
    print("\n Average elapsed time",avg_elapsed_time)
    with open("/home/nsathish/Efficient_object_detection/mmdetection/results/TensorRT_FP16-1080.txt", "a") as myfile:
        myfile.write("Average elapsed time:" + str(avg_elapsed_time) + "\n")        
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results


class MultipleKVAction(argparse.Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary.
    """

    def _is_int(self, val):
        try:
            _ = int(val)
            return True
        except Exception:
            return False

    def _is_float(self, val):
        try:
            _ = float(val)
            return True
        except Exception:
            return False

    def _is_bool(self, val):
        return val.lower() in ['true', 'false']

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for val in values:
            parts = val.split('=')
            key = parts[0].strip()
            if len(parts) > 2:
                val = '='.join(parts[1:])
            else:
                val = parts[1].strip()
            # try parsing val to bool/int/float first
            if self._is_bool(val):
                import json
                val = json.loads(val.lower())
            elif self._is_int(val):
                val = int(val)
            elif self._is_float(val):
                val = float(val)
            options[key] = val
        setattr(namespace, self.dest, options)


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format_only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument(
        '--scale',
        type=float,
        help='To change the input image scale'
    )
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=MultipleKVAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def my_fn():
    local_conf = {
    "checkpoint":'checkpoints/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth', 
    "config":'configs/mask_rcnn_r50_fpn_1x.py', 
    "eval":['bbox', 'proposal'], 
    "format_only":False, 
    "gpu_collect":False, 
    "launcher":'none', 
    "local_rank":0, 
    "options":None, 
    "out":None, 
    "show":False, 
    "tmpdir":None
    }
    parser = argparse.ArgumentParser()
    args=conf =parser.parse_args()
    args.__dict__ = {**args.__dict__, **local_conf}
    return args
def main():
    args = parse_args()
    #scale = args.scale
    #args = my_fn()
    #print(args)
    #pdb.set_trace()
    assert args.out or args.eval or args.format_only or args.show, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results) with the argument "--out", "--eval", "--format_only" '
         'or "--show"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    #cfg.test_pipeline[1]['img_scale'] = (int(640*scale),int(640*scale))
    cfg.data.test['ann_file']="/data/mengtial/COCO/annotations/instances_val2017.json"
    cfg.data.test['img_prefix']="/data/mengtial/COCO/val2017/"
    #print("-------------------------",)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,# Batch size
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model['mask_head']=None
    
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    #print("Model",model)
    fp16_cfg = cfg.get('fp16', None)
    #print("----------------------",fp16_cfg)
    
    if fp16_cfg is not None:
        print("Inside")
        #pdb.set_trace()
        wrap_fp16_model(model)
    
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        
        outputs = single_gpu_test(model, data_loader,args.show )
 
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print('\nwriting results to {}'.format(args.out))
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.options is None else args.options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            dataset.evaluate(outputs, args.eval, **kwargs)


if __name__ == '__main__':
    main()
