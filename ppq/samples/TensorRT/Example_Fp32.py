MODEL          = 'model.onnx'
INPUT_SHAPE    = [1, 3, 224, 224]
SAMPLES        = [torch.rand(size=[INPUT_SHAPE]) for _ in range(256)] # rewirte this to use real data.

# -------------------------------------------------------------------
# 打开 trt_infer 看到具体细节，这个文件是 nvidia 的官方实例
# -------------------------------------------------------------------
from trt_infer import EngineBuilder
builder = EngineBuilder()
builder.create_network('model_fp32.onnx')
builder.create_engine(engine_path='model_fp32.engine', precision="fp16")

# -------------------------------------------------------------------
# 启动 tensorRT 进行推理，你先装一下 trt
# -------------------------------------------------------------------
import tensorrt as trt
import trt_infer

samples = [convert_any_to_numpy(sample) for sample in SAMPLES]
logger = trt.Logger(trt.Logger.INFO)
with open('model_fp32.engine', 'rb') as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

results = []
with engine.create_execution_context() as context:
    inputs, outputs, bindings, stream = trt_infer.allocate_buffers(context.engine)
    for sample in tqdm(samples, desc='TensorRT is running...'):
        inputs[0].host = convert_any_to_numpy(sample)
        [output] = trt_infer.do_inference(
            context, bindings=bindings, inputs=inputs, 
            outputs=outputs, stream=stream, batch_size=1)
        results.append(convert_any_to_torch_tensor(output))
