# 多模型多平台Benchmark

本项目提供了使用PPQ对图像分类、目标检测&实例分割模型在Openvino、TensorRT等多平台上进行量化、推理、精度评估的测试脚本。

项目结构如下：

- **classification**：对图像分类模型的测试benchmark。
- **detection**：对目标检测&实例分割模型的测试benchmark。

- **dynamic_shape_quant**：动态输入的检测模型在TensorRT上的量化推理与精度评估。

使用该项目，请使用以下命令安装以下额外的依赖包

```shell
pip install -r requirements.txt
```

