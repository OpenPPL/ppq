import os
import json
import tensorrt as trt



model_file = "/home/geng/tinyml/ppq/benchmark/dynamic_shape_quant/FP32_model/Retinanet-wo-dynamic-"
onnx_file =  model_file +"FP32" +".onnx"
json_file = model_file + "INT8"+".json"
engine_file = model_file + "INT8"+".engine"

TRT_LOGGER = trt.Logger()

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

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
                value = act_quant[tensor.name]
                tensor_max = abs(value)
                tensor_min = -abs(value)
                tensor.dynamic_range = (tensor_min, tensor_max)
            else:
                print("\033[1;32m%s\033[0m" % tensor.name)


def build_engine(json_file):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    
    config = builder.create_builder_config()


    # ### 如果是动态的onnx模型，则需要加入以下内容
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 3, 150, 220), (1, 3, 480, 640), (1, 3, 650,650))
    config.add_optimization_profile(profile)
    

    parser = trt.OnnxParser(network, TRT_LOGGER)
    config.max_workspace_size = GiB(1)

    if not os.path.exists(onnx_file):
        quit('ONNX file {} not found'.format(onnx_file))

    with open(onnx_file, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    config.set_flag(trt.BuilderFlag.INT8)
    
    setDynamicRange(network, json_file)

    engine = builder.build_engine(network, config)

    with open(engine_file, "wb") as f:
        f.write(engine.serialize())
        

if __name__ == '__main__':
    build_engine(json_file)
    print("\033[1;32mgenerate %s\033[0m" % engine_file)


