import pdb
import pycuda.driver as cuda
import tensorrt as trt
import os
import numpy as np
import pycuda.autoinit
import torch
import pickle
import pdb
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def get_engine(onnx_file_path, engine_file_path=""):
        """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
        TRT_LOGGER = trt.Logger()
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        def build_engine():
            """Takes an ONNX file and creates a TensorRT engine to run inference with"""
            with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
                builder.max_workspace_size = 1 << 28 # 256MiB
                builder.max_batch_size = 1
                #builder.fp16_mode=True
                builder.int8_mode=True
                #pdb.set_trace()
                #print("Builder",builder)
                # Parse model file
                if not os.path.exists(onnx_file_path):
                    print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                    exit(0)
                print('Loading ONNX file from path {}...'.format(onnx_file_path))
                with open(onnx_file_path, 'rb') as model:
                    print('Beginning ONNX file parsing')
                    if not parser.parse(model.read()):
                        print ('ERROR: Failed to parse the ONNX file.')
                        for error in range(parser.num_errors):
                            print (parser.get_error(error))
                        return None
                # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
                network.get_input(0).shape = [1, 3, 512, 512]
                print('Completed parsing of ONNX file')
                print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
                engine = builder.build_cuda_engine(network)
                print("Completed creating Engine")
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
                return engine

        if os.path.exists(engine_file_path):
            # If a serialized engine exists, use it instead of building an engine.
            print("Reading engine from file {}".format(engine_file_path))
            with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            return build_engine()
def allocate_buffers(engine):


    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)

        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    # input_output_path = "/home/nsathish/Efficient_object_detection/mmdetection/buffer/ssd_pickle.dat"
    # buffers=[inputs,outputs,bindings]
    # with open(input_output_path, "wb") as f:
    #     pickle.dump(buffers, f, pickle.HIGHEST_PROTOCOL)
    return inputs, outputs, bindings, stream


def init():
    onnx_file_path = '/home/nsathish/Efficient_object_detection/mmdetection/onnx/ssd512_mmdet.onnx'
    engine_file_path = "/home/nsathish/Efficient_object_detection/mmdetection/tensorrt_engines/ssd512_mmdet_int8.trt"
    global context
    global inputs
    global outputs
    global bindings
    global stream
    global engine
    
    inputs=[]
    outputs=[]
    bindings=[]
    stream = cuda.Stream()
    # with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
    #     inputs, outputs, bindings, stream = allocate_buffers(engine)

    engine = get_engine(onnx_file_path, engine_file_path)
    
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    context = engine.create_execution_context()  #Context is to store intermediate activation values generated during inference