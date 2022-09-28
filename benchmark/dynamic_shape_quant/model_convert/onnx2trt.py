import os
import tensorrt as trt

model_path = "/home/geng/tinyml/ppq/benchmark/dynamic_shape_quant/FP32_model/Retinanet-wo-dynamic-FP32"
onnx_file = model_path+".onnx"
float_engine_file = model_path+".engine"

TRT_LOGGER = trt.Logger()

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def GiB(val):
    return val * 1 << 30


def build_float_engine():
    """build the float32 engine."""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    
    config = builder.create_builder_config()

    # 这里的三个batch必须一致，比如都设置为1，或者都设置为4，都行
    # channel根据网络的通道来设置，一般都为3
    # h和w是可变的，这三个分别是min,opt,max
    # min的意思是最小可支持的尺寸,max是最大可支持的尺寸,opt在min和max之间,表示的应该是最优的h和w
    profile = builder.create_optimization_profile()
    # profile.set_shape("input", (1, 3, 160, 300), (1, 3, 800, 1216), (1, 3, 1220,1220))
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

    engine = builder.build_engine(network, config)

    with open(float_engine_file, "wb") as f:
        f.write(engine.serialize())



if __name__ == '__main__':
    build_float_engine()
