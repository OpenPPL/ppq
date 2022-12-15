import os
import json
import argparse
import tensorrt as trt

TRT_LOGGER = trt.Logger()

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

class ModelData(object):
    DEPLOY_PATH = "deploy.prototxt"
    MODEL_PATH = "model.caffemodel"
    OUTPUT_NAME = "238","251","240","248","250","252"  #caffe2trt needs to specify the name of the output tensor.
    # The original model is a float32 one.
    DTYPE = trt.float32

def GiB(val):
    return val * 1 << 30

def json_load(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
    return data

def setDynamicRange(network, json_file):
    """Sets ranges for network layers."""
    quant_param_json = json_load(json_file)
    act_quant = quant_param_json["act_quant_info"]

    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        if act_quant.__contains__(input_tensor.name):
            print(input_tensor.name)
            value = act_quant[input_tensor.name]
            tensor_max = abs(value)
            tensor_min = -abs(value)
            input_tensor.dynamic_range = (tensor_min, tensor_max)

    for i in range(network.num_layers):
        layer = network.get_layer(i)

        for output_index in range(layer.num_outputs):
            tensor = layer.get_output(output_index)

            if act_quant.__contains__(tensor.name):
                print("\033[1;32mWrite quantization parameters:%s\033[0m" % tensor.name)
                value = act_quant[tensor.name]
                tensor_max = abs(value)
                tensor_min = -abs(value)
                tensor.dynamic_range = (tensor_min, tensor_max)
            else:
                print("\033[1;31mNo quantization parameters are written: %s\033[0m" % tensor.name)


def build_int8_engine(deploy_file, model_file, json_file, engine_file):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    
    config = builder.create_builder_config()

#     # If it is a dynamic onnx model , you need to add the following.
#     # profile = builder.create_optimization_profile()
#     # profile.set_shape("input_name", (batch, channels, min_h, min_w), (batch, channels, opt_h, opt_w), (batch, channels, max_h, max_w)) 
#     # config.add_optimization_profile(profile)
    
    parser = trt.CaffeParser()
    config.max_workspace_size = GiB(1)

    if not os.path.exists(deploy_file):
        quit('deploy_file file {} not found'.format(deploy_file))
    if not os.path.exists(model_file):
        quit('model_file file {} not found'.format(model_file))

    model_tensors = parser.parse(deploy=deploy_file, model=model_file, network=network, dtype=ModelData.DTYPE)
    
    for act_name in ModelData.OUTPUT_NAME:
        network.mark_output(model_tensors.find(act_name))

    config.set_flag(trt.BuilderFlag.INT8)
    
    setDynamicRange(network, json_file)

    engine = builder.build_engine(network, config)

    with open(engine_file, "wb") as f:
        f.write(engine.serialize())


if __name__ == '__main__':
    # Add plugins if needed
    # import ctypes
    # ctypes.CDLL("libmmdeploy_tensorrt_ops.so")
    parser = argparse.ArgumentParser(description='Writing qparams to onnx to convert tensorrt engine.')
    parser.add_argument('--deploy_file', type=str, default=None)
    parser.add_argument('--model_file', type=str, default=None)
    parser.add_argument('--qparam_json', type=str, default=None)
    parser.add_argument('--engine', type=str, default=None)
    arg = parser.parse_args()


    build_int8_engine(arg.deploy_file, arg.model_file, arg.qparam_json, arg.engine)
    print("\033[1;32mgenerate %s\033[0m" % arg.engine)



# python write_qparams_caffe2trt.py --deploy_file=ppq_quant_outputs/quantized.prototxt --model_file
# =ppq_quant_outputs/quantized.caffemodel --qparam_json=ppq_quant_outputs/quant_cfg.json --engine=lidar_int8.engine
