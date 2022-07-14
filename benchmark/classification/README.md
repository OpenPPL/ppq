# Classification Benchmark
对5种模型：Resnet-18, ResNeXt101_64x4d, RegNet_X_1_6GF, ShuffleNetV2_x1_0, MobileNetV2    
在四个平台上:TensorRT, OpenVino, Snpe, Ncnn (gpu, x86cpu, Dsp&Npu, ARM cpu)  
测试四个精度：Onnxruntime FP32，PPQ INT8, QDQ onnxruntime INT8, TargetPlatform INT8 

|model|TargetPlatform|ORT FP32|PPQ INT8|QDQ ORT INT8|RealPlatform INT8|
|----|----|----|----|----|----|
|resnet18|OpenVino|69.764|69.466|69.480|**66.975**|
|resnet18|TRT|69.764|69.578|69.550|69.484|
|resnet18|Snpe|69.764|69.278|69.266|-|
|resnet18|Ncnn|69.764|69.132|69.064|-|
|mobilenetV2|OpenVino|72.017|71.317|71.383|**63.552**|
|mobilenetV2|TRT|72.017|71.413|71.441|71.367|
|mobilenetV2|Snpe|72.017|70.102|70.072|-|
|mobilenetV2|Ncnn|72.017|71.671|71.657|-|
|ResNeXt101_64x4d|OpenVino|82.774|81.978|81.834|**79.297**|
|ResNeXt101_64x4d|TRT|82.985|81.920|--|81.882|
|ResNeXt101_64x4d|Snpe|82.985|81.492|--|-|
|ResNeXt101_64x4d|Ncnn|82.985|82.911|--|-|
|RegNet_X_1_6GF|OpenVino|79.341|78.443|78.593|**72.531**|
|RegNet_X_1_6GF|TRT|79.341|78.531|78.737|78.539|
|RegNet_X_1_6GF|Snpe|79.341|76.616|--|-|
|RegNet_X_1_6GF|Ncnn|79.341|79.243|--|-|
|ShuffleNetV2_x1_0|OpenVino|69.370|68.806|68.802|-|
|ShuffleNetV2_x1_0|TRT|69.370|68.718|--|68.706|
|ShuffleNetV2_x1_0|Snpe|69.370|68.564|--|-|
|ShuffleNetV2_x1_0|Ncnn|69.370|68.748|-|-|


- [x] 完成resnet18在openvino上的前三项精度测试 
- [x] 完成resnet18在四个平台的前三项精度测试
- [x] 完成resnet18在TRT和OpenVino上的部署精度测试
- [ ] 完成5个模型在4个平台上的前三项精度测试
- [ ] 完成5个模型在4个平台上的部署精度测试

2022年7月13日：遇到问题：
- Vit存在无法量化的算子ERF。
- shufflenetV2无法在openvino上推理。
- export_ppq_graph的参数copy_graph=True时，会导致导出的openvino和ORT模型推理精度为0。