import argparse
import os
import struct
import sys
import numpy as np
import tensorrt as trt

INPUT_H = 32
INPUT_W = 32
CHANNEL = 1
OUTPUT_SIZE = 10
INPUT_BLOB_NAME = "input.1"
OUTPUT_BLOB_NAME = "32" 
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def GiB(val):
    return val * 1 << 30

def generateLenetEngine(weight_map, json_file, engine_path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    config = builder.create_builder_config()

    # Adds an input to the network.
    data = network.add_input(INPUT_BLOB_NAME, trt.float32, (1, CHANNEL, INPUT_H, INPUT_W))
    assert data
    data.name = "input.1"

    conv1 = network.add_convolution(input=data,
                                    num_output_maps=6,
                                    kernel_shape=(5, 5),
                                    kernel=weight_map["conv1.weight"],
                                    bias=weight_map["conv1.bias"])

    assert conv1
    conv1.stride = (1, 1)
    conv1.get_output(0).name = "onnx::Relu_11"

    relu1 = network.add_activation(conv1.get_output(0),
                                   type=trt.ActivationType.RELU)
    assert relu1
    relu1.get_output(0).name = "onnx::Pad_12"

    pool1 = network.add_pooling(input=relu1.get_output(0),
                                window_size=trt.DimsHW(2, 2),
                                type=trt.PoolingType.AVERAGE)
    assert pool1
    pool1.stride = (2, 2)
    pool1.get_output(0).name = "onnx::AveragePool_14"


    conv2 = network.add_convolution(pool1.get_output(0), 16, trt.DimsHW(5, 5),
                                    weight_map["conv2.weight"],
                                    weight_map["conv2.bias"])

    assert conv2
    conv2.stride = (1, 1)
    conv2.get_output(0).name = "input"

    relu2 = network.add_activation(conv2.get_output(0),
                                   type=trt.ActivationType.RELU)
    assert relu2
    relu2.get_output(0).name = "onnx::Relu_16"


    pool2 = network.add_pooling(input=relu2.get_output(0),
                                window_size=trt.DimsHW(2, 2),
                                type=trt.PoolingType.AVERAGE)
    assert pool2
    pool2.stride = (2, 2)
    pool2.get_output(0).name = "onnx::Pad_17"

    fc1 = network.add_fully_connected(input=pool2.get_output(0),
                                      num_outputs=120,
                                      kernel=weight_map['fc1.weight'],
                                      bias=weight_map['fc1.bias'])
    assert fc1
    fc1.get_output(0).name = "onnx::AveragePool_19"


    relu3 = network.add_activation(fc1.get_output(0),
                                   type=trt.ActivationType.RELU)
    assert relu3
    relu3.get_output(0).name = "onnx::Relu_27"

    fc2 = network.add_fully_connected(input=relu3.get_output(0),
                                      num_outputs=84,
                                      kernel=weight_map['fc2.weight'],
                                      bias=weight_map['fc2.bias'])
    assert fc2
    fc2.get_output(0).name = "onnx::Gemm_28"

    relu4 = network.add_activation(fc2.get_output(0),
                                   type=trt.ActivationType.RELU)
    assert relu4
    relu4.get_output(0).name = "onnx::Relu_29"

    fc3 = network.add_fully_connected(input=relu4.get_output(0),
                                      num_outputs=OUTPUT_SIZE,
                                      kernel=weight_map['fc3.weight'],
                                      bias=weight_map['fc3.bias'])
    assert fc3
    fc3.get_output(0).name = "onnx::Gemm_30"

    prob = network.add_softmax(fc3.get_output(0))
    assert prob
    prob.get_output(0).name = "32"

    network.mark_output(prob.get_output(0))
    config.max_workspace_size = GiB(1)

    config.set_flag(trt.BuilderFlag.INT8)
    setDynamicRange(network, json_file)

    engine = builder.build_engine(network, config)

    with open(engine_path, "wb") as f:
        f.write(engine.serialize())

    del weight_map
    del engine
    del network
    del builder


def setDynamicRange(network, json_file):
    """Sets ranges for network layers."""
    import json
    with open(json_file) as json_file_info:
        quant_param_json = json.load(json_file_info)

    act_quant = quant_param_json["act_quant_info"]
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        if act_quant.__contains__(input_tensor.name):
            print("\033[1;32mWrite quantization parameters:%s\033[0m" % input_tensor.name)
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


def load_weights(file):
    print(f"Loading weights: {file}")

    assert os.path.exists(file), 'Unable to load weight file.'

    weight_map = {}
    with open(file, "r") as f:
        lines = [line.strip() for line in f]
    count = int(lines[0])
    assert count == len(lines) - 1
    for i in range(1, count + 1):
        splits = lines[i].split(" ")
        name = splits[0]
        cur_count = int(splits[1])
        assert cur_count + 2 == len(splits)
        values = []
        for j in range(2, len(splits)):
            # hex string to bytes to float
            values.append(struct.unpack(">f", bytes.fromhex(splits[j])))
        weight_map[name] = np.array(values, dtype=np.float32)
    return weight_map


if __name__ == '__main__':
    sys.argv[1] # Weight parameter
    sys.argv[2] # Quantization parameter json
    sys.argv[3] # tensorrt engine file
    weight_map = load_weights(sys.argv[1])
    
    generateLenetEngine(weight_map, sys.argv[2], sys.argv[3])