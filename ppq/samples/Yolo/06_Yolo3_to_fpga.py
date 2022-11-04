import torch

from ppq import *
from ppq.api import *
from ppq.quantization.quantizer import FPGAQuantizer

model_path = 'yolo3.onnx' # model path
EXECUTING_DEVICE = 'cuda'
# initialize dataloader, suppose preprocessed input data is in binary format
INPUT_SHAPE = [1, 3, 640, 640]
# npy_array = [np.fromfile(os.path.join(data_path, file_name), dtype=np.float32).reshape(*INPUT_SHAPE) for file_name in os.listdir(data_path)]
dataloader = [torch.zeros(size=INPUT_SHAPE).cuda() for i in range(32)]

# confirm platform and setting
target_platform = TargetPlatform.FPGA_INT8
s = QuantizationSettingFactory.fpga_setting()

class MyFPGAQuantizer(FPGAQuantizer):
    """
    不同厂商的FPGA量化规则不一样，
    你需要自定义量化器，根据硬件执行规则调整量化方案
    """
    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
        config = super().init_quantize_config(operation)
        if operation.type == 'Mul':
            # 取消 Mul 的输入量化
            for input_cfg in config.input_quantization_config:
                input_cfg.state = QuantizationStates.FP32
        return config

    @ property
    def quant_operation_types(self) -> set:
        return {
            'Conv', 'ConvTranspose', 'Gemm', 'Relu', 'PRelu',
            'Clip', 'Pad', 'Resize', 'MaxPool', 'AveragePool',
            'GlobalMaxPool', 'GlobalAveragePool',
            'Mul', 'Add', 'Max', 'Sub', 'Div',
            'LeakyRelu', 'Concat', 'Slice'
        }

# load and schedule graph
ppq_graph_ir = load_onnx_graph(model_path)
ppq_graph_ir = dispatch_graph(ppq_graph_ir, target_platform)

# intialize quantizer and executor
executor = TorchExecutor(ppq_graph_ir, device=EXECUTING_DEVICE)
quantizer = MyFPGAQuantizer(graph=ppq_graph_ir)

with ENABLE_CUDA_KERNEL():
# run quantization
    dummy_input = dataloader[0].to(EXECUTING_DEVICE)    # random input for meta tracing
    quantizer.quantize(
        input_shape=None,
        inputs=dummy_input,                         # some random input tensor, should be list or dict for multiple inputs
        calib_dataloader=dataloader,                # calibration dataloader
        executor=executor,                          # executor in charge of everywhere graph execution is needed
        setting=s,                                  # quantization setting
        calib_steps=32,                             # number of batched data needed in calibration, 8~512
        collate_fn=lambda x: x.to(EXECUTING_DEVICE) # final processing of batched data tensor
    )
    
    graphwise_error_analyse(
        graph=ppq_graph_ir, 
        running_device='cuda', 
        dataloader=dataloader, 
        collate_fn=lambda x: x.to('cuda'))
   
    export_ppq_graph(graph=ppq_graph_ir, 
                     platform=TargetPlatform.ONNXRUNTIME, 
                     graph_save_to='Quantized.onnx')
