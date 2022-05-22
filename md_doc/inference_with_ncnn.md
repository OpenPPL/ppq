# Inference with ncnn
this tutorial gives you a simple illustration how you could actually use PPQ to quantize your model and export quantization parameter file to inference with ncnn as your backend. Note that ncnn supports conversion of both caffe and onnx models, in this tutorial we take the onnx model, shufflenet-v2, as an example to illustrate the whole process going from a ready-to-quantize model to int8 ready-to-run artifact which can be executed in quantization mode by ncnn.

## Optimize Your Network
ncnn does not support some shape-related operations like Shape, Gather ... so before we send the network
to PPQ for quantization, we need to eliminate these operations, for onnx, we can use onnx-simplifier to
simplify our model
```shell
python -m onnxsim --input-shape=1,3,224,224 shufflenet-v2.onnx shufflenet-v2-sim.onnx
```
the command above will make shapes constant, so operations supporting dynamic shapes can be eliminated by
constant folding.

## Quantize Your Network
as we have specified in [how_to_use](./how_to_use.md), we should prepare our calibration dataloader, confirm
the target platform on which we want to deploy our model(*TargetPlatform.NCNN_INT8* in this case), load our
simplified model, initialize quantizer and executor, and then run the quantization process
```python
import os
import numpy as np
import torch
from ppq.api import load_onnx_graph
from ppq.api.interface import dispatch_graph, QUANTIZER_COLLECTION
from ppq.core import TargetPlatform
from ppq.executor import TorchExecutor
from ppq import QuantizationSettingFactory

model_path = '/models/shufflenet-v2-sim.onnx' # onnx simplified model
data_path  = '/data/ImageNet/calibration' # calibration data folder
EXECUTING_DEVICE = 'cuda'

# initialize dataloader 
INPUT_SHAPE = [1, 3, 224, 224]
npy_array = [np.fromfile(os.path.join(data_path, file_name), dtype=np.float32).reshape(*INPUT_SHAPE) for file_name in os.listdir(data_path)]
dataloader = [torch.from_numpy(np.load(npy_tensor)) for npy_tensor in npy_array]

# confirm platform and setting
target_platform = TargetPlatform.NCNN_INT8
setting = QuantizationSettingFactory.academic_setting() # for ncnn, no fusion

# load and schedule graph
ppq_graph_ir = load_onnx_graph(model_path)
ppq_graph_ir = dispatch_graph(ppq_graph_ir, target_platform, setting)

# intialize quantizer and executor
executor = TorchExecutor(ppq_graph_ir, device='cuda')
quantizer = QUANTIZER_COLLECTION[target_platform](graph=ppq_graph_ir)

# run quantization
calib_steps = max(min(512, len(dataloader)), 8)     # 8 ~ 512
dummy_input = dataloader[0].to(EXECUTING_DEVICE)    # random input for meta tracing
quantizer.quantize(
        inputs=dummy_input,                         # some random input tensor, should be list or dict for multiple inputs
        calib_dataloader=dataloader,                # calibration dataloader
        executor=executor,                          # executor in charge of everywhere graph execution is needed
        setting=setting,                            # quantization setting
        calib_steps=calib_steps,                    # number of batched data needed in calibration, 8~512
        collate_fn=lambda x: x.to(EXECUTING_DEVICE) # final processing of batched data tensor
)

# export quantization param file and model file
export_ppq_graph(graph=ppq_ir_graph, platform=TargetPlatform.NCNN_INT8, graph_save_to='shufflenet-v2-sim-ppq', config_save_to='shufflenet-v2-sim-ppq.table')
```
note that your dataloader should provide batch data which is in the same shape of the input of simplified model, because
simplified model can't take dynamic-shape inputs.

## Convert Your Model
if you have compiled ncnn correctly, there should be executables in the installation binary folder which can convert onnx model
to ncnn binary format
```shell
/path/to/your/ncnn/build/install/bin/onnx2ncnn shufflenet-v2-sim-ppq.onnx shufflenet-v2-ncnn.param shufflenet-v2-ncnn.bin
```
nothing would print if everything goes right, then we may use ncnn optimize tool to optimize the converted model
```shell
/path/to/your/ncnn/build/install/bin/ncnnoptimize shufflenet-v2-ncnn.param shufflenet-v2-ncnn.bin shufflenet-v2-ncnn-opt.param shufflenet-v2-ncnn-opt.bin 0
```
then along with ppq generated quantization table, we may convert the ncnn optimized model into the final int8 version 
```shell
/path/to/your/ncnn/build/install/bin/ncnn2int8 shufflenet-v2-ncnn-opt.param shufflenet-v2-ncnn-opt.bin shufflenet-v2-ncnn-int8.param shufflenet-v2-ncnn-int8.bin shufflenet-v2-sim-ppq.table
```
there should be final files generated in your current folder
```
|--working
      |-- shufflenet-v2-ncnn-int8.param
      |-- shufflenet-v2-ncnn-int8.bin
```

## Write Script and Run Inference
suppose you get a sample image in the your folder and you want to run inference to see if the model has been quantized correctly,
you need to load your image into ncnn Mat and pre-process it.
```c++
{
    const char* imagepath = "/workspace/sample.jpg";
    cv::Mat bgr = cv::imread(imagepath, 1);
    ncnn::Net shufflenetv2;

    shufflenetv2.opt.light_mode = true;
    shufflenetv2.opt.num_threads = 4; // omp thread num, for cpu inference only

    shufflenetv2.load_param("/workspace/shufflenet-v2-ncnn-int8.param");
    shufflenetv2.load_model("/workspace/shufflenet-v2-ncnn-int8.bin");

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 224, 224);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = shufflenetv2.create_extractor();

    ex.input("data", in); // input name

    ncnn::Mat out;
    ex.extract("fc", out); // output name
}
```
and if you want to obtain the final probability and print top-5 predictions
```c++
{
    std::vector<float> cls_scores;
    // softmax -> probs
    { 
        ncnn::Layer* softmax = ncnn::create_layer("Softmax");

        ncnn::ParamDict pd;
        softmax->load_param(pd);

        softmax->forward_inplace(out, shufflenetv2.opt);

        delete softmax;
    }
    out = out.reshape(out.w * out.h * out.c);

    cls_scores.resize(out.w);
    for (int j = 0; j < out.w; j++)
    {
        cls_scores[j] = out[j];
    }
    // print top-5
    int size = cls_scores.size();
    std::vector<std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int> >());

    // print topk and score
    for (int i = 0; i < 5; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
    }
}
```
note that if you are using gpu which is supported by vulkan, you just need to correctly install vulkan, compile ncnn with 
vulkan option enabled, and turn on vulkan switch after initializing the network
```c++
ncnn::Net shufflenetv2;
shufflenetv2.opt.use_vulkan_compute = true;
```
then ncnn will run the quantized binary model on gpu.