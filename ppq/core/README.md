## PPQ.Core(PPQ 核心定义)
你正浏览 PPQ 的核心数据结构定义，本目录中的文件描述了 PPQ 软件的底层逻辑：

1. ppq.core.common: ppq 预定义常量，用户可以修改该文件从而配置软件的相应功能。

2. ppq.core.config: ppq 基础定义，包括版本号，软件名等内容。

3. ppq.core.data: ppq 基础数据类型，包含 pytorch.Tensor, numpy.ndarray 的数据类型转换。

4. ppq.core.defs: ppq 元类型及全局工具函数定义。

5. ppq.core.ffi: ppq 应用程序编程接口，包含调用 c/c++, cuda 代码的相关逻辑。

6. ppq.core.quant: ppq 核心量化数据结构定义【非常重要】。

7. ppq.core.storage: ppq 持久化操作的相关定义。

## TensorQuantizationConfig(Tensor 量化控制结构体)
PPQ 使用量化控制结构体描述量化行为，该结构体被定义在 ppq.core.quant 中。截止 PPQ 0.6.6 版本，该结构体由 15 项不同的属性组成。我们将向你介绍这一核心数据结构体的设计构想。

### 1. QuantizationPolicy 量化策略
在 TensorQuantizationConfig 当中，首当其冲地内容是 TQC.policy，这是一个 QuantizationPolicy 对象。
policy 属性用于描述量化的规则，一个完整的量化策略是由多个量化属性(QuantizationProperty)组合完成的；在 PPQ 中目前我们支持 8 种不同的量化属性，你可以使用以下属性来组合形成自定义的量化规则:

1. PER_TENSOR: 以 Tensor 为单位完成量化，每个 Tensor 使用一个 scale 和 offset 信息。

2. PER_CHANNEL: 以 Channel 为单位完成量化，每个 Channel 使用一个 scale 和 offset 信息。

3. LINEAR: 线性量化，通常的 INT8, INT16 皆属于线性量化，在线性量化的表示中不存在指数位。

4. FLOATING: 浮点量化，包括 FP8 E4M3, FP8 E5M2, FP16, BF16 等格式，在浮点量化中数据由底数和指数两部分组成。

5. SYMMETRICAL: 对称量化，在量化计算中不启用 offset。

6. ASYMMETRICAL: 非对称量化，在量化计算中启用 offset 完成量化偏移。

7. POWER_OF_2: 限制 scale 取值必须为 2 的整数次幂，这一量化行为多见于端侧以及浮点量化。

8. DYNAMIC: 启用动态量化策略，对于每一批次的数据，scale 与 offset 都将被动态地计算更新。

下图解释了浮点量化与线性量化的区别：

![image](https://user-images.githubusercontent.com/43309460/199235366-1e83ed97-0731-4e1d-abeb-b7121e3d2a94.png)

### 2. 线性量化与相关属性

线性量化允许与下列属性进行组合：

    QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL,
    QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR,
    QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL,
    QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR,
    QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.POWER_OF_2,
    QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.POWER_OF_2,
    QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.POWER_OF_2,
    QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.POWER_OF_2,

线性量化是最为常用的数值量化方法，有些时候我们也称其为均匀量化，在线性量化中，量化操作的计算方法为：

- Unscaled FP32 = (FP32 / scale) - offset
- INT8 = Clip(Round(Unscale FP32), quant_min, quant_max)
- Dequantized FP32 = (INT8 + offset) * scale

其中 Round 函数行为由 TQC.rounding(RoundingPolicy) 属性确定，PPQ 支持 7 种不同的取整策略，其中 ROUND_HALF_EVEN 是最常见的取整策略，关于取整策略的详细讨论可以参考 https://en.wikipedia.org/wiki/Rounding

quant_min, quant_max 分别由 TQC.quant_min, TQC.quant_max 属性确定，对于线性量化而言他们是整数，通常为[-128, 127]。部分框架使用 [-127, 127] 作为截断值，在部分场景下如此定义将有优势，但在 Onnx 的 Q/DQ 算子定义中不允许使用 [-127, 127] 作为截断。

PPQ 可以模拟 1-32 bit 的任意位宽量化，但若以部署为目的，不建议使用 8 bit 之外的配置。用户须知高位宽量化可能造成 Scale 过小，以至于浮点下溢出。

### 3. 浮点量化与相关属性

浮点量化允许与下列属性进行组合：

    QuantizationProperty.SYMMETRICAL | QuantizationProperty.FLOATING | QuantizationProperty.PER_CHANNEL | QuantizationProperty.POWER_OF_2,
    QuantizationProperty.SYMMETRICAL | QuantizationProperty.FLOATING | QuantizationProperty.PER_TENSOR | QuantizationProperty.POWER_OF_2,

在浮点量化中，量化函数的计算方法为：

- Unscaled FP32 = (FP32 / scale)
- FP8 = Convert(Unscale FP32, quant_min, quant_max)
- Dequantized FP32 = FP8 * scale

其中 Convert 函数行为复杂，其转换过程分为三种不同的情况：
- 当 Unscaled FP32 大于 quant_max，或者小于 quant_min，则直接进行截断
- 当 Unscaled FP32 幅值大于 FP8 能够表达的最小值，此时需要移去多余的底数位，并对底数进行四舍五入
- 当 Unscaled FP32 数据小于规范化 FP8 能够表达的最小值，此时浮点下溢出，此时我们计算 FP8 = Round(Unscaled FP32 / FP8_min) * FP8_min

其中 FP8_min 是非规格化 FP8 能够表达的最小值。对于 FP8 E4M3 标准而言，其能表示的最大值为 448.0，最小值为 -448.0。

quant_min, quant_max 分别由 TQC.quant_min, TQC.quant_max 属性确定，对于 FLOATING 量化，我们引入一个新的属性 TQC.exponent_bits(int)。使用这个属性来指定总位宽中有多少数位用于表示指数(相应地，底数位为总位宽-指数位-1)。

在浮点量化中，尺度因子的选取对量化效果的影响不大，因此用户可以使用 constant 校准策略(见 ppq.quantization.observer)将所有尺度因子设置为1。

关于浮点量化的具体细节可以参考 [本文](https://zhuanlan.zhihu.com/p/574825662)

### 4. 其他量化控制属性

1. TQC.num_of_bits(int)：量化位宽，对于 INT8, FP8 量化，量化位宽为 8。对于 INT16, FP16 量化，量化位宽为16。

2. TQC.state(QuantizationStates): 量化状态，在 PPQ 中目前有共计 8 种不同的量化状态，该属性极大地丰富了 PPQ 量化信息的语义，使得我们能够更加灵活地控制量化行为。该属性可以被用于切换 量化 / 非量化 状态；执行量化联合定点；执行参数烘焙。

3. TQC.channel_axis(int): 量化轴，对于 PER_CHANNEL 量化，使用这个属性来指定沿着那一维度展开量化，如执行 Per-tensor 量化，该属性被忽略，用户可以将其设置为 None。

4. TQC.observer_algorithm(str): observer 算法，其中 observer 是用于确定 scale 和 offset 的对象，使用这个属性指明要使用何种类型的 observer 确定 scale 和 offset

5. TQC.dominator(TensorQuantizationConfig): 一个指向父量化信息的指针。在 PPQ 中 TQC 与 TQC 之间并不是独立的，他们之间可以存在父子关系。所有子量化信息与父量化信息共享 scale 和 offset

6. TQC.visiblity(QuantizationVisibility): 导出可见性，使用这个属性来告知 ppq 的导出器是否需要导出当前的 TQC。

### 5. 量化控制结构体的初始化

TensorQuantizationConfig 是 PPQ 中的核心数据结构，它总是由 Quantizer 对象完成创建的：

    # 下面这段代码为一个指定的算子创建了相应的 Tensor Quantization Config
    quantizer = PFL.Quantizer(platform=TargetPlatform.TRT_FP8, graph=graph) # 取得 TRT_FP8 所对应的量化器
    quantizer.quantize_operation(op_name = op.name, platform = dispatching[op.name])

在 PPQ 当中，Quantizer 的职责即是为算子初始化他们的量化控制结构体。不同的量化器将按照不同的规则创建控制结构体，如 TRT_FP8 所对应的量化器 只会为了 Conv, Gemm 算子创建量化信息，要求他们的输入按照对称-浮点-Per Channel的方式完成量化。而 DSP_INT8 所对应的量化器为几乎所有算子创建量化信息，要求他们按照非对称-线性-Per Tensor的方式完成量化。

用户可以手动创建量化控制结构体，使用 ppq.lib 中的接口：

    # 创建一个默认的线性量化控制结构体(对称, per-tensor)
    from ppq.lib import LinearQuantizationConfig
    TQC = LinearQuantizationConfig()

    # 创建一个默认的浮点量化控制结构体(FP8 E4M3)
    from ppq.lib import FloatingQuantizationConfig
    TQC = FloatingQuantizationConfig()

### 6. 量化控制结构体的校准

绝大部分的 TensorQuantizationConfig 在完成初始化之后都无法使用-他们的 scale 与 offset 均为空值，且 Quantizer 在初始化他们时会将其状态(TQC.state)置为 INITIAL，处于这个状态的量化信息在计算过程中不会被启用。

我们必须送入一定量数据，进行必要 Calibration 操作后才能为网络中的量化信息确定合理的 scale 与 offset 值，这一过程是由种类繁多的 Observer 完成的：

    # PPQ 目前支持 8 种不同的 Observer
    OBSERVER_TABLE = {
        'minmax': TorchMinMaxObserver,
        'kl': TorchHistObserver,
        'percentile': TorchPercentileObserver,
        'mse': TorchMSEObserver,
        'isotone': TorchIsotoneObserver,
        'constant': ConstantObserver,
        'floating': DirectMSEObserver,
        'isotone': ...
    }

这些 Observer 会负责在网络计算过程中收集必要的统计信息，并为 TQC 的 scale 与 offset 赋予有效的值。在完成一切之后，Observer 还会负责将 TQC 的状态(TQC.state)修改为 ACTIVED。此时量化信息将被正式启用，从而在网络前向传播模拟量化计算。

关于 Observer 的讨论，可以参考 [本视频](https://www.bilibili.com/video/BV1QF41157aM)

### 7. 量化控制结构体的父子链接

在我们讨论量化时，对于那些存在着多个输入的算子，例如 add, concat，它们的所有输入总是被要求有着相同的 scale。为了表述这种语义，我们为 TQC 添加了 TQC.dominator 属性，这一属性可以指向另一个量化控制结构体。

假设我们存在两个不同的量化控制结构体 A, B：

- 语句 A.dominator = B 表示 A 将共享 B 的 scale 与 offset(A.policy, A.num_of_bits等属性仍将使用自己的)。于此同时 A.state 将被修改为 OVERLAPPED(A 将不再启用)
- 语句 A.master = B 表示 A 将共享 B 的 scale 与 offset(A.policy, A.num_of_bits等属性仍将使用自己的)。于此同时 A.state 将被修改为 PASSIVE(A 将仍然启用，但不具有独立的量化参数)

如果 A 已经是其他量化结构体 C 的父节点，则上述过程将级联地使得 B 成为 A, C 共同的父节点，A, C 都将共享 B 的 scale 与 offset。

下图简述了在量化控制结构体的生命周期中，量化状态是如何变迁的（[量化优化过程](https://github.com/openppl-public/ppq/tree/master/ppq/quantization/optim)将负责修改量化控制信息的状态）：

![Quantization State](https://user-images.githubusercontent.com/43309460/199236632-ec69ca29-9900-4875-8299-a196546d0dde.png)

## PPQ.Core.Common(PPQ 全局控制常量)
PPQ 全局控制常量被定义在 ppq.core.common.py 文件中，用户可以修改其中的定义以调整 PPQ 程序功能。截止 PPQ 0.6.6 版本，共计 38 个常量可以被设置。
这些常量影响网络解析过程，校准过程与导出逻辑。我们在这里向你阐述部分常用的修改项：

1. OBSERVER_MIN_SCALE - 校准过程 Scale 的最小值，迫于实际部署的需要，Int8 量化的尺度因子不能过小，否则将导致浮点下溢出等数值问题。这个参数决定了所有 PPQ Calibratior 可以提供的最小尺度因子值。该参数只影响校准，不影响训练过程（训练过程仍然可能产生较小的Scale）。该参数不适用于 FP8 量化。

2. OBSERVER_KL_HIST_BINS - 校准过程中 kl 算法的参数，该参数影响 kl 算法的效果，用户可以调整其为 1024, 2048, 4096, 8192 或其他。

3. OBSERVER_PERCENTILE - 校准过程中 Percentile 算法的参数，该参数影响 percentile 算法的效果，用户可以调整其为 0.999, 0.9995, 0.9999, 0.99995, 0.99999 或其他。

4. OBSERVER_MSE_HIST_BINS - 校准过程中 mse 算法的参数，该参数影响 mse 算法的效果，用户可以调整其为 1024, 2048, 4096, 8192 或其他。

5. FORMATTER_FORMAT_CONSTANT_INPUT - 读取 Onnx 图时，是否将图中所有以 Constant 算子作为输入的参数转换为 Constant Variable，部分推理引擎不识别 Constant 算子，PPQ 的优化过程也没有对 Constant 算子进行适配（特指那些卷积的参数是以 Constant 算子输入的情况），因此建议开启该选项。

6. FORMATTER_FUSE_BIAS_ADD - 读取 Onnx 图时，是否尝试合并 Bias Add。部分情况下，导出 Onnx 时前端框架会把 Conv, ConvTranspose, Gemm 等层的 Bias 单独拆分成一个 Add 算子，这将造成后续处理逻辑的错误，因此建议开启该选项。

7. FORMATTER_REPLACE_BN_TO_CONV - 是否将单独的 Batchnorm 替换为卷积，这可能导致错误，因为我们无法判断 Batchnorm 的维数，PPQ 会将所有孤立的（无法进行 BN-Fusion 的） Batchnorm 替换为 Conv2d，对于三维或一维网络而言，这可能会导致错误的结果。 

8. FORMATTER_REMOVE_IDENTITY - 是否移除图中所有的 Identity 算子。

9. FORMATTER_REMOVE_ISOLATED - 是否移除图中的所有孤立的节点。

10. PASSIVE_OPERATIONS - 一个集合，如果一个算子的类型出现在该集合中，PPQ 对该算子执行输入-输出联合定点。即保证该算子的输入变量与输出变量共享 Scale，如果算子有多个输入-输出变量，该操作只影响第一个输入和第一个输出变量。