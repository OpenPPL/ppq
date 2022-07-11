# Classification Benchmark
对5种模型：Resnet-18, ResNeXt-101, SE-ResNet-50, ShuffleNetV2, MobileNetV2    
在四个平台上:TensorRT, OpenVino, Snpe, Ncnn (gpu, x86cpu, Dsp&Npu, ARM cpu)  
测试四个精度：Onnxruntime FP32，PPQ INT8, QDQ onnxruntime INT8, TargetPlatform INT8 

|model|TargetPlatform|ORT FP32|PPQ INT8|QDQ ORT INT8|RealPlatform INT8|
|----|----|----|----|----|----|
|resnet18|OpenVino|69.764|69.466|**67.109**|**66.985**|
|resnet18|TRT|69.764|69.548|69.524|69.534|
|resnet18|Snpe|69.764|69.278|69.266|-|
|resnet18|Ncnn|69.764|69.132|69.062|-|
|mobilenetV2|OpenVino|71.88|--|--|-|
|mobilenetV2|TRT|71.88|--|--|-|
|mobilenetV2|Snpe|71.88|--|--|-|
|mobilenetV2|Ncnn|71.88|--|--|-|

- [x] 完成resnet18在openvino上的前三项精度测试 
- [x] 完成resnet18在四个平台的前三项精度测试
- [x] 完成resnet18在TRT和OpenVino上的部署精度测试
- [ ] 完成5个模型在4个平台上的前三项精度测试
- [ ] 完成5个模型在4个平台上的部署精度测试
