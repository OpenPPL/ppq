# Deploy Model with APIs that come with TensorRT
This project describes how to use tensorrt's own api combined with ppq to build an int8 quantization model


## Environment setup


#### Use Docker Image
- Docker image is recommended:

```bash
docker pull stephen222/ppq:ubuntu18.04_cuda11.4_cudnn8.4_trt8.4.1.5
```

- Create a docker container:

```bash
docker run -it --rm --ipc=host --gpus all --mount type=bind,source=your custom path,target=/workspace stephen222/ppq:ubuntu18.04_cuda11.4_cudnn8.4_trt8.4.1.5 /bin/bash
```

#### Build trt environment

If you want to install it, we strongly suggest you install TensorRT through [tar file](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-841/install-guide/index.html#installing-tar)

- After installation, you'd better add TensorRT environment variables to bashrc by:

  ```bash
  cd ${TENSORRT_DIR} # To TensorRT root directory
  echo '# set env for TensorRT' >> ~/.bashrc
  echo "export TENSORRT_DIR=${TENSORRT_DIR}" >> ~/.bashrc
  echo 'export LD_LIBRARY_PATH=$TENSORRT_DIR/lib:$TENSORRT_DIR' >> ~/.bashrc
  source ~/.bashrc
  ```

## Prepare data and models to quantify your network

**Steps**:

- Please refer to the script `ProgramEntrance.py`
- WORKING_DIRECTORY is the directory where you store data and models, and the quantized results will also be exported to this directory.
- Create the folder WORKING_DIRECTORY , e.g. `working`, and create the folder `data` in the WORKING_DIRECTORY folder to store data, data can be `.bin` files or `.npy` files arranged in (1,c,h,w) format.(Note that the data must be preprocessed)
- Change the name of your model to `model.onnx`, then put it in the WORKING_DIRECTORY folder.
- TargetPlatform should select `TargetPlatform.TRT_INT8`.
- MODEL_TYPE choose `NetworkFramework.ONNX`.
- NETWORK_INPUTSHAPE fill in the shape of the data, e.g. `[1, 3, 224, 224]`.
- CALIBRATION_BATCHSIZE is the batch during optimization, it is recommended to set it to 16 or 32 if your computing platform has enough computing power, otherwise, it can also be set to 1.
- If the last layer of your model is a plugin operator, such as `yolo`, `nms`, etc., please add the following code to the `ProgramEntrance.py` script. The following code uses `yolo` as an example. These three shapes: [1, 36, 19, 19]，[1, 36, 38, 38], [1, 36, 76, 76] correspond to the three outputs of the model.

```bash
def happyforward(*args, **kwards):
    return torch.zeros([1, 36, 19, 19]).cuda(), torch.zeros([1, 36, 38, 38]).cuda(), torch.zeros([1, 36, 76, 76]).cuda()
register_operation_handler(happyforward, 'yolo', platform=TargetPlatform.FP32)
```
- Other parameters are default.
- Run script `python ProgramEntrance.py`
- After the script is executed, you will get 3 files in your working directory, `quantized.onnx`, `quant_cfg.json`, `quantized.wts`.
- `quantized.onnx` is is better for quantization that is used to deploy.
- `quant_cfg.json` contains quantization parameters.
- `quantized.wts` contains quantized weight parameters, if you want to deploy with trt.OnnxParser, please ignore it.


**Note**:

- `quantized.onnx` may change slightly relative to the original onnx model structure, for example, bn and conv are merged, so the tensorrt api should also be adjusted
- If you want to quantify a dynamic onnx model, you firstly need to change its inputs and outputs to static.


## Convert and Quantify your model

- Here is an example of the lenet model

Please refer to the lenet network demo [Creating A Network Definition of TensorRT API](https://github.com/openppl-public/ppq/tree/master/ppq/samples/TensorRT/lenet_demo).

- Get the lenet's onnx model
```bash
python generate_onnx.py
```

- In order to run this demo, you can randomly generate 200-300 `.bin` data, but if in actual deployment, you need to use preprocessed real data


**Python API**:

- Please note that the name of the network input and the name of the op need to be consistent with the name in `quant_cfg json`, as shown below:

```bash
data.name = "input.1"
conv1.get_output(0).name = "onnx::Relu_11"
relu1.get_output(0).name = "onnx::Pad_12"
pool1.get_output(0).name = "onnx::AveragePool_14"
conv2.get_output(0).name = "input"
relu2.get_output(0).name = "onnx::Relu_16"
pool2.get_output(0).name = "onnx::Pad_17"
fc1.get_output(0).name = "onnx::AveragePool_19"
relu3.get_output(0).name = "onnx::Relu_27"
fc2.get_output(0).name = "onnx::Gemm_28"
fc3.get_output(0).name = "onnx::Gemm_30"
prob.get_output(0).name = "32"
```

- Then run the `lenet_int8.py` to generate int8-trt engine.

```bash
python lenet_int8.py quantized.wts quant_cfg.json lenet_int8.engine
```

**C++ API**:

- Please note that the name of the network input and the name of the op need to be consistent with the name in `quant_cfg json`, as shown below:
```bash
conv1->getOutput(0)->setName("onnx::Relu_11");
relu1->getOutput(0)->setName("onnx::Pad_12");
pool1->getOutput(0)->setName("onnx::AveragePool_14");
conv2->getOutput(0)->setName("input");
relu2->getOutput(0)->setName("onnx::Relu_16");
pool2->getOutput(0)->setName("onnx::Pad_17");
fc1->getOutput(0)->setName("onnx::AveragePool_19");
fc2->getOutput(0)->setName("onnx::Gemm_28");
relu4->getOutput(0)->setName("onnx::Relu_29");
fc3->getOutput(0)->setName("onnx::Gemm_30");
```

- Compile the lenet demo and run
```
mkdir build&&cd build
cmake ..
make -j$(nproc)   eg: make -j8 , make -j6
./bin/lenet_int8 quantized.wts quant_cfg.json lenet_int8.engine
```

## Run Inference
Here is an inference script of tensorrt engine

```python
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    def __repr__(self):
        return self.__str__()

def alloc_buf_N(engine,data):
    """Allocates all host/device in/out buffers required for an engine."""
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    data_type = []
    for binding in engine:
        if engine.binding_is_input(binding):
            size = data.shape[0]*data.shape[1]*data.shape[2]*data.shape[3]
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            data_type.append(dtype)
            # Allocate memory on the host
            host_mem = cuda.pagelocked_empty(size, dtype)
            # Allocate memory on the device
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, data_type[0])
            device_mem = cuda.mem_alloc(host_mem.nbytes)            
            bindings.append(int(device_mem))
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def do_inference_v2(context, inputs, bindings, outputs, stream, data):
    """
    Inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    for inp in inputs:
        cuda.memcpy_htod_async(inp.device, inp.host, stream)
    context.set_binding_shape(0, data.shape)
    # Run inference.
    context.execute_async(batch_size=1, bindings=bindings, stream_handle=stream.handle)

    # Transfer predictions back from the GPU.
    for out in outputs:
        cuda.memcpy_dtoh_async(out.host, out.device, stream)

    # Writes the contents of the system buffers back to disk to ensure data synchronization.
    stream.synchronize()

    # Return only the host outputs.
    return [out.host for out in outputs] 

trt_logger = trt.Logger(trt.Logger.INFO)

def load_engine(engine_path):
    print("\033[1;32musing %s\033[0m" % engine_path)
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

if __name__ == '__main__':

    src_data = np.fromfile(bin_file,dtype=np.float32)
    inputs = np.array(src_data).astype(np.float32)
    data = np.reshape(inputs,(batch,channel,height,width))
    engine = load_engine(engine_path)
    context = engine.create_execution_context()
    inputs_alloc_buf, outputs_alloc_buf, bindings_alloc_buf, stream_alloc_buf = alloc_buf_N(engine,data)
    inputs_alloc_buf[0].host = np.ascontiguousarray(inputs)

    trt_feature = do_inference_v2(context, inputs_alloc_buf, bindings_alloc_buf, outputs_alloc_buf,stream_alloc_buf, data)
    trt_feature = np.asarray(trt_feature)
    print(trt_feature)
```

## If there is a bad case of the model accuracy dropping, please submit an issue in the ppq community
