# Dynamic Shape Quantization

本项目实现了一个使用`动态输入`的检测模型使用PPQ量化并在TensorRT上推理的demo。

本仓库用到的模型也可以通过[网盘获取](https://pan.baidu.com/s/1D1xZuQN6bR221u_NkmHPpA )(提取码：s655)

### 1. 模型量化

使用PPQ量化入口脚本`ProgramEntrance.py`，设置目标平台为`TargetPlatform.TRT_INT8`，将导出一个新的FP32 onnx模型以及其存有其量化参数的json文件。

除此之外，也可以尝试使用TensorRT原生工具量化FP32 onnx模型，相关脚本在`./tensorrt_quant`下，用来对比PPQ的量化效果。

### 2.模型转换

接下来我们需要将onnx模型转为TensorRT平台的engine模型，同时将量化参数写入模型。

 在`./model_convert`下提供了两个转换类型：

`onnx2trt.py`提供了任意动态onnx模型转为TensorRT模型的方法，只需提供一个onnx模型即可输出一个engine模型。

`quant_onnx2trt.py`提供了将量化参数写入onnx模型并转为TensorRT模型的方法，需要提供一个onnx模型和PPQ导出的量化参数json文件，能够输出一个量化后的engine模型。

由于待推理模型的输入是动态的，因此在转换之前，请统计或者预估出输入图片可能的动态shape范围，并修改对应的如下位置的代码。min的意思是最小可支持的尺寸,max是最大可支持的尺寸,opt在min和max之间,表示的应该是最优的h和w

```python
# modify in onnx2trt.py and quant_onnx2trt.py
profile.set_shape("input", (1, 3, min_h, min_w), (1, 3, opt_h, opt_w, (1, 3, max_h,max_w))
```

### 3. 模型推理

本项目以不带后处理的RetinaNet-wo检测模型为例，编写了动态推理的过程，具体实现在。

由于TensoRT在推理前需要给模型每个input和output分配显存，因此我们必须要确定模型在动态推理时，模型的input和output的shape, 相关代码在`inference_dynamic.py`。这一部分代码需要针对模型特定结构单独设计。

```python
# 确定output的shape,和input相关
binding_shapes = {}
h,w = input_tensor.shape[2:]
h,w = h//8,w//8
for i in range(1,6):
    binding_shapes[f"output{i}"] = (1,720,h,w)
    binding_shapes[f"output{i+5}"] = (1,36,h,w)
    h = math.ceil(h/2)
    w = math.ceil(w/2)
```

### 4. 精度评估

在`dynamic_trt_retinanet-wo_test.py`中实现了对RetinaNet-wo检测模型的动态推理和精度评估。

|    model     | Precision | map   | Input Size | Input type | FPS   |
| :----------: | --------- | ----- | ---------- | ---------- | ----- |
| Retinanet-wo | FP32      | 26.8% | (480,640)  | Dynamic    | 11.91 |
| Retinanet-wo | PPQ-INT8  | 26.9% | (480,640)  | Dynamic    | 16.20 |
| Retinanet-wo | TRT-INT8  | 26.9% | (480,640)  | Dynamic    | 15.30 |

* Test paltform is TensorRT with 2080ti 

### 5. 执行流程总结

```shell
python ProgramEntrance.py #使用PPQ获取模型量化参数json文件
python onnx2trt.py        #获取FP32的engine模型（可选）
python quant_onnx2trt.py #将量化参数写入模型并转为INT8 engine模型
python dynamic_trt_retinanet-wo_test.py #评估FP32模型或者INT8模型精度
```



