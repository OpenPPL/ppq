# TensorRT Quantization

这个仓库提供了使用TensorRT原生量化工具获取INT8模型的方法，该脚本支持动态输入模型的量化。

首先我们需要有一个FP32 onnx模型，以及用于量化的校准数据集。

修改配置文件`config.json`

```json
{
    "onnx_file": "models/Retinanet-wo-dynamic-FP32.onnx",  //FP32模型路径
    "int8_trt_engine_file": "engines/Retinanet-wo-dynamic-INT8.engine", //导出的INT8模型路径
    "calibration_table": "tables/Retinanet-wo-dynamic-FP32.table", //导出的量化参数表路径
    "calibrate_dir": "calibrate-data", //校准数据集路径，需要文件是.bin格式
    "onnx_model_batchSize": 1,
    "width": 640,   //校准数据集图片的宽度
    "height": 480,  //校准数据集图片的高度
    "channel": 3,
    "net-type" : "onnx"
}
```

修改`onnx2trt_quantization.py`中动态推理的图片范围。min的意思是最小可支持的尺寸,max是最大可支持的尺寸,opt在min和max之间,表示的应该是最优的h和w。

```python
profile.set_shape("input", (1, 3, min_h, min_w), (1, 3, opt_h, opt_w, (1, 3, max_h,max_w))
```

执行脚本，完成量化。

```shell
python onnx2trt_quantization.py
```

