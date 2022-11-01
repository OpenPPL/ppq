## TensorQuantizationConfig(Tensor 量化控制结构体)
PPQ 使用量化控制结构体描述量化行为，该结构体被定义在 ppq.core.quant 中。截止 PPQ 0.6.6 版本，该结构体由 15 项不同的属性组成。本文将向你介绍这一核心数据结构体的设计构想。

### QuantizationPolicy 量化策略
在 TensorQuantizationConfig 当中，首当其冲地内容是 TQC.policy，这是一个 QuantizationPolicy 对象。
policy 属性用于描述量化的规则，一个完整的量化策略是由多个量化属性(QuantizationProperty)组合完成的；在 PPQ 中目前我们支持 8 种不同的量化属性，你可以使用以下属性来组合形成自定义的量化规则:
  - PER_TENSOR: 以 Tensor 为单位完成量化，每个 Tensor 使用一个 scale 和 offset 信息。
  - PER_CHANNEL: 以 Channel 为单位完成量化，每个 Channel 使用一个 scale 和 offset 信息。
  - LINEAR: 线性量化，通常的 INT8, INT16 皆属于线性量化，在线性量化的表示中不存在指数位。
  - FLOATING: 浮点量化，包括 FP8 E4M3, FP8 E5M2, FP16, BF16 等格式，在浮点量化中数据由底数和指数两部分组成。
  - SYMMETRICAL: 对称量化，在量化计算中不启用 offset。
  - ASYMMETRICAL: 非对称量化，在量化计算中启用 offset 完成量化偏移。
  - POWER_OF_2: 限制 scale 取值必须为 2 的整数次幂，这一量化行为多见于端侧以及浮点量化。
  - DYNAMIC: 启用动态量化策略，对于每一批次的数据，scale 与 offset 都将被动态地计算更新。

下图解释了浮点量化与线性量化的区别：

![image](https://user-images.githubusercontent.com/43309460/199235366-1e83ed97-0731-4e1d-abeb-b7121e3d2a94.png)

### 线性量化与相关属性

线性量化允许与下列属性进行组合：

    QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL,
    QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR,
    QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL,
    QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR,
    QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.POWER_OF_2,
    QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.POWER_OF_2,
    QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.POWER_OF_2,
    QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.POWER_OF_2,

在线性量化中，量化函数的计算方法为：

- Unscaled FP32 = (FP32 / scale) - offset
- INT8 = Clip(Round(Unscale FP32), quant_min, quant_max)
- Dequantized FP32 = (INT8 + offset) * scale

其中 Round 函数行为由 TQC.rounding(RoundingPolicy) 属性确定，PPQ 支持 7 种不同的取整策略，其中 ROUND_HALF_EVEN 是最常见的取整策略，关于取整策略的详细讨论可以参考 https://en.wikipedia.org/wiki/Rounding

quant_min, quant_max 分别由 TQC.quant_min, TQC.quant_max 属性确定，对于线性量化而言他们是整数，通常为[-128, 127]

### 浮点量化与相关属性

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

其中 FP8_min 是非规格化 FP8 能够表达的最小值。

quant_min, quant_max 分别由 TQC.quant_min, TQC.quant_max 属性确定，对于浮点量化而言他们是浮点数，通常为[-448.0, +448.0]，对于 FLOATING 量化，我们引入一个新的属性 TQC.exponent_bits(int)。使用这个属性来指定总位宽中有多少数位用于表示指数(相应地，底数位为总位宽-指数位-1)。

关于浮点量化的具体细节可以参考 [本文](https://zhuanlan.zhihu.com/p/574825662)

### 其他属性
  - TQC.num_of_bits(int)：量化位宽，对于 INT8, FP8 量化，量化位宽为 8。对于 INT16, FP16 量化，量化位宽为16。
  - TQC.state(QuantizationStates): 量化状态，在 PPQ 中目前有共计 8 种不同的量化状态，该属性极大地丰富了 PPQ 量化信息的语义，使得我们能够更加灵活地控制量化行为。该属性可以被用于切换 量化 / 非量化 状态；执行量化联合定点；执行参数烘焙。
  - TQC.channel_axis(int): 量化轴，对于 PER_CHANNEL 量化，使用这个属性来指定沿着那一维度展开量化
  - TQC.observer_algorithm(str): observer 算法，其中 observer 是用于确定 scale 和 offset 的对象，使用这个属性指明要使用何种类型的 observer 确定 scale 和 offset
  - TQC.dominator(TensorQuantizationConfig): 一个指向父量化信息的指针。在 PPQ 中 TQC 与 TQC 之间并不是独立的，他们之间可以存在父子关系。所有子量化信息与父量化信息共享 scale 和 offset
  - TQC.visiblity(QuantizationVisibility): 导出可见性，使用这个属性来告知 ppq 的导出器是否需要导出当前的 TQC。

### 量化控制结构体的初始化

TensorQuantizationConfig 是 PPQ 中的核心数据结构，它总是由 Quantizer 对象完成创建的：

    # 下面这段代码为一个指定的算子创建了相应的 Tensor Quantization Config
    quantizer = PFL.Quantizer(platform=TargetPlatform.TRT_FP8, graph=graph) # 取得 TRT_FP8 所对应的量化器
    quantizer.quantize_operation(op_name = op.name, platform = dispatching[op.name])

在 PPQ 当中，Quantizer 的职责即是为算子初始化他们的量化控制结构体。不同的量化器将按照不同的规则创建控制结构体，如 TRT_FP8 所对应的量化器 只会为了 Conv, Gemm 算子创建量化信息，要求他们的输入按照对称-浮点-Per Channel的方式完成量化。而 DSP_INT8 所对应的量化器为几乎所有算子创建量化信息，要求他们按照非对称-线性-Per Tensor的方式完成量化。

### 量化控制结构体的校准

绝大部分的 TensorQuantizationConfig 在完成初始化之后都无法使用-他们的 scale 与 offset 均为空值，且 Quantizer 在初始化他们时会将其状态(TQC.state)置为 INITIAL，处于这个状态的量化信息在计算过程中不会被启用。

我们必须送入一定量数据，进行必要 Calibration 操作后才能为网络中的量化信息确定合理的 scale 与 offset 值，这一过程是由种类繁多的 Observer 完成的：

    # PPQ 目前支持 7 种不同的 Observer
    OBSERVER_TABLE = {
        'minmax': TorchMinMaxObserver,
        'kl': TorchHistObserver,
        'percentile': TorchPercentileObserver,
        'mse': TorchMSEObserver,
        'isotone': TorchIsotoneObserver,
        'constant': ConstantObserver,
        'floating': DirectMSEObserver
    }

这些 Observer 会负责在网络计算过程中收集必要的统计信息，并为 TQC 的 scale 与 offset 赋予有效的值。在完成一切之后，Observer 还会负责将 TQC 的状态(TQC.state)修改为 ACTIVED。此时量化信息将被正式启用，从而在网络前向传播模拟量化计算。

关于 Observer 的讨论，可以参考 [本视频](https://www.bilibili.com/video/BV1QF41157aM)

### 量化控制结构体的父子链接

在我们讨论量化时，对于那些存在着多个输入的算子，例如 add, concat，它们的所有输入总是被要求有着相同的 scale。为了表述这种语义，我们为 TQC 添加了 TQC.dominator 属性，这一属性可以指向另一个量化控制结构体。

假设我们存在两个不同的量化控制结构体 A, B：

- 语句 A.dominator = B 表示 A 将共享 B 的 scale 与 offset(A.policy, A.num_of_bits等属性仍将使用自己的)。于此同时 A.state 将被修改为 OVERLAPPED(A 将不再启用)
- 语句 A.master = B 表示 A 将共享 B 的 scale 与 offset(A.policy, A.num_of_bits等属性仍将使用自己的)。于此同时 A.state 将被修改为 PASSIVE(A 将仍然启用，但不具有独立的量化参数)

如果 A 已经是其他量化结构体 C 的父节点，则上述过程将级联地使得 B 成为 A, C 共同的父节点，A, C 都将共享 B 的 scale 与 offset。

下图展示了在量化控制结构体的生命周期中，量化状态是如何变迁的：

![Quantization State](https://user-images.githubusercontent.com/43309460/199236632-ec69ca29-9900-4875-8299-a196546d0dde.png)

