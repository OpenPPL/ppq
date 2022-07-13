# Classification Benchmark
对5种模型：Resnet-18, ResNeXt101_64x4d, EfficientNet-b7, ShuffleNetV2_x1_0, MobileNetV2    
在四个平台上:TensorRT, OpenVino, Snpe, Ncnn (gpu, x86cpu, Dsp&Npu, ARM cpu)  
测试四个精度：Onnxruntime FP32，PPQ INT8, QDQ onnxruntime INT8, TargetPlatform INT8 

|model|TargetPlatform|ORT FP32|PPQ INT8|QDQ ORT INT8|RealPlatform INT8|
|----|----|----|----|----|----|
|resnet18|OpenVino|69.764|69.466|**67.109**|**66.985**|
|resnet18|TRT|69.764|69.548|69.524|69.534|
|resnet18|Snpe|69.764|69.278|69.266|-|
|resnet18|Ncnn|69.764|69.132|69.062|-|
|mobilenetV2|OpenVino|72.017|--|--|-|
|mobilenetV2|TRT|72.017|--|--|-|
|mobilenetV2|Snpe|72.017|--|--|-|
|mobilenetV2|Ncnn|72.017|--|--|-|
|ResNeXt101_64x4d|OpenVino|82.985|--|--|-|
|ResNeXt101_64x4d|TRT|82.985|--|--|-|
|ResNeXt101_64x4d|Snpe|82.985|--|--|-|
|ResNeXt101_64x4d|Ncnn|82.985|--|--|-|
|Vit_B_16|OpenVino|81.074|--|--|-|
|Vit_B_16|TRT|81.074|--|--|-|
|Vit_B_16|Snpe|81.074|--|--|-|
|Vit_B_167|Ncnn|81.074|--|--|-|
|ShuffleNetV2_x1_0|OpenVino|69.370|--|--|-|
|ShuffleNetV2_x1_0|TRT|69.370|--|--|-|
|ShuffleNetV2_x1_0|Snpe|69.370|--|--|-|
|ShuffleNetV2_x1_0|Ncnn|69.370|--|--|-|

- [x] 完成resnet18在openvino上的前三项精度测试 
- [x] 完成resnet18在四个平台的前三项精度测试
- [x] 完成resnet18在TRT和OpenVino上的部署精度测试
- [ ] 完成5个模型在4个平台上的前三项精度测试
- [ ] 完成5个模型在4个平台上的部署精度测试
