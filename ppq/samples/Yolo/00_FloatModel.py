ONNX_PATH        = 'models/yolov5s.v5.onnx'       # 你的模型位置
ENGINE_PATH      = 'Output/yolov5s.v5(fp32).engine' # 生成的 Engine 位置

# -------------------------------------------------------------------
# 打开 trt_infer 看到具体细节，这个文件是 nvidia 的官方实例
# -------------------------------------------------------------------
from trt_infer import EngineBuilder
builder = EngineBuilder()
builder.create_network(ONNX_PATH)
builder.create_engine(engine_path=ENGINE_PATH, precision="fp32")
