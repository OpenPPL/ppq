# Inference with PPL CUDA
this tutorial gives you a simple illustration how you could actually use PPQ to quantize your model and export
quantization parameter file to inference with ppl cuda as your backend. Similar to [inference_with_ncnn](./inference_with_ncnn.md), we use an onnx model, shufflenet-v2, as an example here to illustrate the whole process
going from ready-to-quantize model to ready-to-deploy polished onnx model, with quantization parameter file generated

## Quantize Your Network
as we have specified in [how_to_use](./how_to_use.md), we should prepare our calibration dataloader, confirm
the target platform on which we want to deploy our model(*TargetPlatform.PPL_CUDA_INT8* in this case), load our
ready-to-quantize model, initialize quantizer and executor, and then run the quantization process
```python
import torch
from ppq.api import load_onnx_graph, export_ppq_graph
from ppq.api.interface import dispatch_graph, QUANTIZER_COLLECTION
from ppq.core import TargetPlatform
from ppq.executor import TorchExecutor
from ppq import QuantizationSettingFactory

model_path = '/models/shufflenet-v2.onnx'
data_path  = '/data/ImageNet/calibration'
EXECUTING_DEVICE = 'cuda'

# initialize dataloader, suppose preprocessed input data is in binary format
INPUT_SHAPE = [1, 3, 224, 224]
npy_array = [np.fromfile(os.path.join(data_path, file_name), dtype=np.float32).reshape(*INPUT_SHAPE) for file_name in os.listdir(data_path)]
dataloader = [torch.from_numpy(np.load(npy_tensor)) for npy_tensor in npy_array]

# confirm platform and setting
target_platform = TargetPlatform.PPL_CUDA_INT8
setting = QuantizationSettingFactory.pplcuda_setting()

# load and schedule graph
ppq_graph_ir = load_onnx_graph(model_path)
ppq_graph_ir = dispatch_graph(ppq_graph_ir, target_platform, setting)

# intialize quantizer and executor
executor = TorchExecutor(ppq_graph_ir, device=EXECUTING_DEVICE)
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
export_ppq_graph(graph=ppq_graph_ir, platform=target_platform, graph_save_to='shufflenet-v2-ppq', config_save_to='shufflenet-v2-ppq.json')
```

## Write Script and Inference
suppose we have a sample image in our working space, and we have quantization param file and model exported by
PPQ, now we are going to run quantization inference on the OpenPPL GPU INT8 backend
```
|--workspace
      |-- shufflenet-v2-ppq.onnx
      |-- shufflenet-v2-ppq.json
      |-- sample.jpg
```
we need to preprocess the sample image, here is an opencv example
```c++
// convert bgr to rgb and normalizes
void preprocess(const cv::Mat& src_img, float* in_data)
{
    const int32_t height = src_img.rows;
    const int32_t width = src_img.cols;

    cv::Mat rgb_img;
    // bgr -> rgb
    cv::cvtColor(src_img, rgb_img, cv::COLOR_BGR2RGB);
    vector<cv::Mat> rgb_channels(3);
    cv::split(rgb_img, rgb_channels);

    cv::Mat r_channel_fp32(height, width, CV_32FC1, in_data + 0 * height * width);
    cv::Mat g_channel_fp32(height, width, CV_32FC1, in_data + 1 * height * width);
    cv::Mat b_channel_fp32(height, width, CV_32FC1, in_data + 2 * height * width);
    vector<cv::Mat> rgb_channels_fp32{r_channel_fp32, g_channel_fp32, b_channel_fp32};

    // convert uint8 to fp32, y = (x - mean) / std
    const float mean[3] = {123.675f, 116.28f, 103.53f}; // change mean & std according to your dataset & training param
    const float std[3] = {58.395f, 57.12f, 57.375f};
    for (uint32_t i = 0; i < rgb_channels.size(); ++i) {
        rgb_channels[i].convertTo(rgb_channels_fp32[i], CV_32FC1, 1.0f / std[i], -mean[i] / std[i]);
    }
}
```
the function above normalizes source image *src_img* and store it in *in_data* in CHW memory layout, which can be used
as direct input passing to PPL backend.

then we need to register a cuda engine for execution, providing options such as which device to use and the memory policy.
```c++
{
    //create engine pool
    vector<unique_ptr<Engine>> engines;
    //choose option and create engine
    cuda::EngineOptions options;
    options.device_id = 0; // gpu device
    options.mm_policy = cuda::MM_COMPACT; // memory save

    cuda::RegisterBuiltinOpImpls();
    //create cuda engine and enable quick-select algorithm
    //will choose optimal kernels for Conv and Gemm
    auto cuda_engine = cuda::EngineFactory::Create(options);
    cuda_engine->Configure(cuda::ENGINE_CONF_USE_DEFAULT_ALGORITHMS, true);
}
```
and we need to load the quantization file, if you want to run in fp16 mode, you can skip this part
```c++
{
    const char* quantization_file = "/workspace/shufflenet-v2-ppq.json";
    string file_content;
    ifstream ifile;
    ifile.open(quantization_file, ios_base::in);

    stringstream ss;
    ss << ifile.rdbuf();
    file_content = ss.str();
    ifile.close();

    cuda_engine->Configure(cuda::ENGINE_CONF_SET_QUANT_INFO, file_content.c_str());
}
```
and in order to let OpenPPL make further option, we need to pass the input shape to the engine
```c++
{
    vector<vector<int64_t>> input_shapes;
    // we only have one input here
    input_shapes.push_back(vector<int64_t>({1, src_img.channels(), src_img.rows, src_img.cols}));
    vector<utils::Array<int64_t>> dims(input_shapes.size());
    for (uint32_t i = 0; i < input_shapes.size(); ++i) {
        auto& arr = dims[i];
        arr.base = input_shapes[i].data();
        arr.size = input_shapes[i].size();
    }
    cuda_engine->Configure(cuda::ENGINE_CONF_SET_INPUT_DIMS, dims.data(), dims.size());
    engines.emplace_back(cuda_engine);
}
```
of course we also need to load the onnx model
```c++
{
    //create onnx runtime && load onnx model
    const char* onnx_model = "/workspace/shufflenet-v2-ppq.onnx";
    //create onnx runtime builder
    auto builder = unique_ptr<onnx::RuntimeBuilder>(onnx::RuntimeBuilderFactory::Create());
    builder->LoadModel(onnx_model);
}
```
and then we need to collect and manage resources
```c++
{
    // collect all engines resgistered
    vector<Engine*> engine_ptrs(engines.size());
    for (uint32_t i = 0; i < engines.size(); ++i) {
        engine_ptrs[i] = engines[i].get();
    }
    
    // manage engine resources
    onnx::RuntimeBuilder::Resources resources;
    resources.engines = engine_ptrs.data();
    resources.engine_num = engine_ptrs.size();

    builder->SetResources(resources);
    //do preparation work
    builder->Preprocess();
}
```
finally, we are able to create PPL runtime instance, send preprocessed input and do the execution
```c++
{
    //load image, resize and preprocess into a buffer
    cv::Mat src_img = cv::imread("/workspace/sample.jpg", 1);
    cv::resize(src_img, src_img, cv::Size(224,224));
    vector<float> in_data_(src_img.rows * src_img.cols * src_img.channels());
    preprocess(src_img, in_data_.data());
    
    //create ppl runtime
    unique_ptr<Runtime> runtime;
    runtime.reset(builder->CreateRuntime());

    //allocate buffer for input tensor
    auto input_tensor = runtime->GetInputTensor(0);
    const std::vector<int64_t> input_shape = {1, src_img.channels(), src_img.rows, src_img.cols};
    input_tensor->GetShape()->Reshape(input_shape);
    input_tensor->ReallocBuffer();

    // set input data descriptor & copy input data
    TensorShape src_desc = *input_tensor->GetShape();
    src_desc.SetDataType(DATATYPE_FLOAT32);
    src_desc.SetDataFormat(DATAFORMAT_NDARRAY);

    input_tensor->ConvertFromHost(in_data_.data(), src_desc);
    
    // run inference
    runtime->Run();
}
```
to obtain the output result and see top-5 predictions
```c++
{
    // get output tensor and element size
    auto output_tensor = runtime->GetOutputTensor(0);
    uint64_t output_size = output_tensor->GetShape()->GetElementsExcludingPadding();
    // copy output tensor to host buffer
    std::vector<float> output_data_(output_size);
    float* output_data = output_data_.data();
    // set output data descriptor & extract output
    TensorShape dst_desc = *output_tensor->GetShape();
    dst_desc.SetDataType(DATATYPE_FLOAT32);
    dst_desc.SetDataFormat(DATAFORMAT_NDARRAY);
    output_tensor->ConvertToHost(output_data, dst_desc);
    
    // sort and print out top-5 prediction
    vector<pair<float, uint64_t>> arr;
    for(uint64_t i = 0; i < output_size; i++)
    {
        arr.emplace_back(output_data[i], i);
    }
    auto cmp = [](const pair<float, uint64_t>& p1, const pair<float, uint64_t>& p2) -> bool
    {
        return p1.first > p2.first;
    };
    sort(arr.begin(), arr.end(), cmp);
    
    for(uint64_t i = 0; i < 5; i++)
    {
        fprintf(stderr, "%ld %f\n", arr[i].second, arr[i].first);
    }
}
```
