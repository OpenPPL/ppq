## Project hierarchy 代码结构

* IR - PPQ 量化计算图定义，以及图上相关操作(算子变换, 算子融合等)，量化计算图是基于 onnx 标准的
* api - 用户接口，包含基本 api 函数
* core - 核心数据结构定义、全局常量定义、编程语言接口等
* executor - PPQ 训练与推理引擎，用于执行 PPQ IR
* parser - 网络读取与导出模块
* quantization - 量化逻辑
  * algorithm - 算法相关逻辑
  * analyse - 量化误差分析工具
  * measure - 损失函数集合
  * observer - 量化校准算法集合
  * optim - 量化优化过程集合
  * qfunction - PPQ 核心量化函数
  * quantizer - 量化器集合
* samples - 示例文件
* scheduler - 调度器
* utils - 工具函数
* csrc - C++ / Cuda 高性能算子库

## Reading Recommendations  推荐阅读
* core.quant - 核心量化结构抽象
* core.common - 全局常量定义
* IR.search - 图模式匹配库
* IR.quantize - 量化图定义
* executor.torch - 量化推理引擎
* quantization.optim - 量化优化过程
* quantization.analyse - 量化误差分析
* quantization.quantizer - 量化器
* scheduler.perseus - 调度器
* utils.round - 量化取整策略
* csrc - 高性能算子库
