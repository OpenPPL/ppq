# Deploy Model with TensorRT
This document describes the quantization deployment process of the TensorRT and how PPQ writes quantization parameters to the tensorrt engine.


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

- Please refer to the script `ProgramEntrance.py`.
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
- `quantized.wts` contains quantized weight parameters, if you want to deploy with trt.OnnxParser, please ignore it. But if you want to deploy with the api that comes with tensorrt, please refer to [Define the model directly using the TensorRT API](https://github.com/openppl-public/ppq/tree/master/md_doc/deploy_trt_by_api.md). 


**Note**:

- If you want to quantify a dynamic onnx model, you firstly need to change its inputs and outputs to static.


## Convert and Quantify your model

Please refer to the script [Convert onnx to int8-engines](https://github.com/openppl-public/ppq/tree/master/ppq/utils/write_qparams_onnx2trt.py).

```bash
python write_qparams_onnx2trt.py \
    --onnx=quantized.onnx \
    --qparam_json=quant_cfg.json \
    --engine=int8-trt.engine
```

## Run Inference
Here is an inference script of tensorrt

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
