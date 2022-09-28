# Deploy Model with TensorRT
This document describes the quantization deployment process of the TensorRT and how PPQ writes quantization parameters to the onnx and convert to int8-trt engine.


## Environment setup

### Use Docker Image
- Docker image is recommended:

```bash
docker pull stephen222/ppq:ubuntu18.04_cuda11.4_cudnn8.4_trt8.4.1.5
```

- Create a docker container:

```bash
docker run -it --rm --ipc=host --gpus all --mount type=bind,source=your custom path,target=/workspace stephen222/ppq:ubuntu18.04_snpe1.65 /bin/bash
```

### You can also install it yourself

If you want to install it, we strongly suggest you install TensorRT through [tar file](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-841/install-guide/index.html#installing-tar)

- After installation, you'd better add TensorRT environment variables to bashrc by:

  ```bash
  cd ${TENSORRT_DIR} # To TensorRT root directory
  echo '# set env for TensorRT' >> ~/.bashrc
  echo "export TENSORRT_DIR=${TENSORRT_DIR}" >> ~/.bashrc
  echo 'export LD_LIBRARY_PATH=$TENSORRT_DIR/lib:$TENSORRT_DIR' >> ~/.bashrc
  source ~/.bashrc
  ```


## Quantize Your Network
as we have specified in [how_to_use](./how_to_use.md), we should prepare our calibration dataloader, confirm
the target platform on which we want to deploy our model(*TargetPlatform.TRT_INT8* in this case), load our
simplified model, initialize quantizer and executor, and then run the quantization process


**Note**:

- If you want to quantify a dynamic onnx model, you firstly need to change its inputs and outputs to static.


**Steps**:

- Please refer to the script ProgramEntrance.py, just change the variable TARGET_PLATFORM to TargetPlatform.TRT_INT8
- Create a working directory, e.g. `working`, then create the `data` folder in the working directory as the quantitative dataset path, change the name of the onnx model to model.onnx store it in the working directory.
- Run script `python ProgramEntrance.py`
- After the script is executed, you will get two files in your working directory, `quantized.onnx`, `quant_cfg.json`
- If you want to quantify a dynamic onnx model, you can change `quantized.onnx` to static 
- Finally, follow the steps below to write the quantization parameters `quant_cfg.json` into the onnx model `quantized.onnx`.


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
