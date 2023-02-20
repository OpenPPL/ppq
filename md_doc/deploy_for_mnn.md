# Deploy Model with MNN
This document describes the quantization deployment process of the MNN and how PPQ writes quantization parameters to the mnn model.

## Environment setup

#### Use Docker Image
- Docker image is recommended:

```bash
docker pull stephen222/ppq:mnn_0219_update
```

- Create a docker container:

```bash
docker run -it --rm --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -e NVIDIA_DRIVER_CAPABILITIES=compute,utility --gpus all --mount type=bind,source=用户自定义路径,target=/workspace stephen222/ppq:mnn_0219_update /bin/bash
```

## Prepare data and models to quantify your network

**Steps**:

- Please refer to the script `ProgramEntrance.py`.
- WORKING_DIRECTORY is the directory where you store data and models, and the quantized results will also be exported to this directory.
- Create the folder WORKING_DIRECTORY , e.g. `working`, and create the folder `data` in the WORKING_DIRECTORY folder to store data which can end with `.bin` files or `.npy` arranged in (1,c,h,w) format. (Note that the data must be preprocessed)
- Change the name of your model to `model.onnx`(or `model.caffemodel`, `model.prototxt` for caffe), then put it in the WORKING_DIRECTORY folder.
- TargetPlatform should select `TargetPlatform.MNN_INT8`.
- MODEL_TYPE choose `NetworkFramework.ONNX` or `NetworkFramework.CAFFE`.
- NETWORK_INPUTSHAPE fill in the shape of the data, e.g. `[1, 3, 224, 224]`.
- CALIBRATION_BATCHSIZE is the batch during optimization, it is recommended to set it to 16 or 32 if your computing platform has enough computing power, otherwise, it can also be set to 1.
- Other parameters are default.
- Run script `python ProgramEntrance.py`
- After the script is executed, you will get files in your working directory, `quantized.onnx`, `quant_cfg.json`(or `quantized.caffemodel`, `quantized.prototxt` and `quant_cfg.json`).
- `quant_cfg.json` contains quantization parameters.

for example:
```python
WORKING_DIRECTORY = 'working'                             # choose your working directory
TARGET_PLATFORM   = TargetPlatform.MNN_INT8          # choose your target platform
MODEL_TYPE        = NetworkFramework.ONNX                 # or NetworkFramework.CAFFE
INPUT_LAYOUT          = 'chw'                             # input data layout, chw or hwc
NETWORK_INPUTSHAPE    = [1, 3, 224, 224]                  # input shape of your network
CALIBRATION_BATCHSIZE = 1                                # batchsize of calibration dataset
EXECUTING_DEVICE      = 'cuda'                            # 'cuda' or 'cpu'.
REQUIRE_ANALYSE       = False
DUMP_RESULT           = False
TRAINING_YOUR_NETWORK = True                              # 是否需要 Finetuning 一下你的网络
```

## Convert Your Model(take resnet18 as an example)
Put the quantized quant_cfg.json and quantized.onnx under the /work path in the container

for onnx:
```shell
./MNNConvert -f ONNX --modelFile quantized.onnx --MNNModel resnet18.mnn --bizCode mnn
```

for caffe:
```shell
./MNNConvert -f CAFFE --modelFile quantized.caffemodel --prototxt quantized.prototxt  --MNNModel resnet18.mnn --bizCode mnn
```

## Quant Your Model(take resnet18 as an example)
```shell
./quantized.out resnet18.mnn  resnet18_quant.mnn quant_cfg.json 
```
The quantized mnn model can be obtained: resnet18_quant.mnn

## It is recommended to use the inference tool we compiled, because sometimes the inference tool will affect the inference results
## ./inference resnet18_quant.mnn data.bin
## inference result: output.bin
## Note that data.bin should be preprocessed data

## If there is a bad case of the model accuracy dropping, please contact us.

