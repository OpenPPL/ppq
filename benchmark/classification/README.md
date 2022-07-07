# Classification Benchmark
对5种模型：Resnet-18, ResNeXt-101, SE-ResNet-50, ShuffleNetV2, MobileNetV2    
在四个平台上:TensorRT, OpenVino, Snpe, Ncnn (gpu, x86cpu, Dsp&Npu, ARM cpu)  
测试四个精度：Onnxruntime FP32，PPQ INT8, QDQ onnxruntime INT8, TargetPlatform INT8 

- [x] 完成resnet18在openvino上的前三项精度测试 
- [ ] 完成resnet18在四个平台的前三项精度测试
- [ ] 完成5个模型在4个平台上的前三项精度测试
- [ ] 完成5个模型在4个平台上的部署精度测试
