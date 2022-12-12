# Deploy Model with SNPE DSP
This document describes the quantization deployment process of the SNPE DSP and how PPQ writes quantization parameters to the SNPE model.
 

## Environment setup
Refer to [Qualcomm official documentation](https://developer.qualcomm.com/sites/default/files/docs/snpe/setup.html) to configure the Linux host environment. The SNPE model conversion and quantization are all done on the Linux host. SNPE supports reading Caffe and Onnx models. This document uses the ONNX model as an example.

## Quantize Your Network
as we have specified in [how_to_use](./how_to_use.md), we should prepare our calibration dataloader, confirm
the target platform on which we want to deploy our model(*TargetPlatform.QNN_DSP_INT8* in this case), load our
simplified model, initialize quantizer and executor, and then run the quantization process
```python
import os

import numpy as np
import torch

from ppq import QuantizationSettingFactory
from ppq.api import dispatch_graph, export_ppq_graph, load_onnx_graph
from ppq.core import TargetPlatform
from ppq.executor import TorchExecutor
from ppq.lib import Quantizer

model_path = '/models/shufflenet-v2-sim.onnx' # onnx simplified model
data_path  = '/data/ImageNet/calibration' # calibration data folder
EXECUTING_DEVICE = 'cuda'

# initialize dataloader 
INPUT_SHAPE = [1, 3, 224, 224]
npy_array = [np.fromfile(os.path.join(data_path, file_name), dtype=np.float32).reshape(*INPUT_SHAPE) for file_name in os.listdir(data_path)]
dataloader = [torch.from_numpy(np.load(npy_tensor)) for npy_tensor in npy_array]

# confirm platform and setting
target_platform = TargetPlatform.QNN_DSP_INT8
setting = QuantizationSettingFactory.dsp_setting()

# load and schedule graph
ppq_graph_ir = load_onnx_graph(model_path)
ppq_graph_ir = dispatch_graph(ppq_graph_ir, target_platform)

# intialize quantizer and executor
executor = TorchExecutor(ppq_graph_ir, device='cuda')
quantizer = Quantizer(graph=ppq_graph_ir, platform=target_platform)

# run quantization
calib_steps = max(min(512, len(dataloader)), 8)     # 8 ~ 512
dummy_input = dataloader[0].to(EXECUTING_DEVICE)    # random input for meta tracing
ppq_graph_ir = quantizer.quantize(
        inputs=dummy_input,                         # some random input tensor, should be list or dict for multiple inputs
        calib_dataloader=dataloader,                # calibration dataloader
        executor=executor,                          # executor in charge of everywhere graph execution is needed
        setting=setting,                            # quantization setting
        calib_steps=calib_steps,                    # number of batched data needed in calibration, 8~512
        collate_fn=lambda x: x.to(EXECUTING_DEVICE) # final processing of batched data tensor
)

# export quantization param file and model file
export_ppq_graph(graph=ppq_graph_ir, platform=TargetPlatform.QNN_DSP_INT8, graph_save_to='shufflenet-v2-sim-ppq', config_save_to='shufflenet-v2-sim-ppq.table')```
```

## Convert Your Model
The snpe-onnx-to-dlc tool converts the ppq export onnx model to an equivalent DLC representation.
```shell
snpe-onnx-to-dlc -i ppq_export_fp32.onnx -o fp32.dlc
```
Generate 8 or 16 bit TensorFlow style fixed point weight and activations encodings for a floating point SNPE model.
The snpe-dlc-quantize tool converts non-quantized DLC models into quantized DLC models.
```shell
snpe-dlc-quantize --input_dlc fp32.dlc --input_list path_to_binary_calidata --output_dlc quant.dlc
```

Finally, write the PPQ quantization parameters to quant.dlc. We have fully tested the script in snpe version 1.43. In recent SNPE releases, if the option –quantization_overrides is provided during model conversion the user can provide a json file with parameters to use for quantization. These will be cached along with the model and can be used to override any quantization data carried from conversion (eg TF fake quantization) or calculated during the normal quantization process in snpe-dlc-quantize. 

```shell
python3 write_qparams_to_snpe_dlc.py --input_dlc_model quant.dlc --qparam quant.json
```

## Run Inference
Model inference using mobile dsp. The inputs must be presented as a tensor of shape (batch x height x width x channel)

```shell
snpe-net-run --container ppq_export_quant.dlc --input_list path_to_data --use_dsp
```
