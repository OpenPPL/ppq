import json
import copy
from ppq.core import *


class GraphFormatSetting():
    def __init__(self) -> None:
        # 有一些平台不支持Constant Op，这个pass会尝试将 Constant Operation 的输入转变为 Parameter Variable
        # Some deploy platform does not support constant operation, 
        # this pass will convert constant operation into parameter variable(if can be done).
        self.format_constant_op = True
        
        # 融合Conv和Batchnorm
        # Fuse Conv and Batchnorm Layer. This pass is necessary and curicial.
        self.fuse_conv_bn       = True
        
        # 将所有的parameter variable进行分裂，使得每个variable具有至多一个输出算子
        # Split all parameter variables, making all variable has at most 1 output operation. 
        # This pass is necessary and curicial.
        self.format_paramters   = True
        
        # 一个你必须要启动的 Pass
        # This pass is necessary and curicial.
        self.format_cast        = True
        
        # 尝试从网络中删除所有与输出无关的算子和 variable
        # Remove all unnecessary operations and variables(which are not link to graph output) from current graph,
        # notice that some platfrom use unlinked variable as quantization parameters, do not set this as true if so.
        self.delete_isolate     = True


class SSDEqualizationSetting():
    def __init__(self) -> None:
        # Equalization 优化级别，目前只支持level 1，对应 Conv--Relu--Conv 和 Conv--Conv 的拉平
        # optimization level, only support level 1 for now
        # you shouldn't modify this
        self.opt_level            = 1
        
        # 在计算scale的时候，所有低于 channel_ratio * max(W) 的值会被裁减到 channel_ratio * max(W)
        # channel ratio used to calculate equalization scale
        # all values below this ratio of the maximum value of corresponding weight
        # will be clipped to this ratio when calculating scale
        self.channel_ratio        = 0.5

        # loss的降低阈值，优化后的loss低于 原来的loss * 降低阈值, 优化才会发生
        # optimized loss must be below this threshold of original loss for algo to take effect
        self.loss_threshold       = 0.8
        
        # 是否对权重进行正则化
        # whether to apply layer normalization to weights
        self.layer_norm           = False

        # 算法迭代次数，3次对于大部分网络足够
        # num of iterations, 3 would be enough for most networks
        # it takes about 10 mins for one iteration
        self.iteration            = 3


class EqualizationSetting():
    def __init__(self) -> None:
        # Equalization 优化级别，如果选成 1 则不进行多分支拉平，如果选成 2，则进行跨越 add, sub 的多分支拉平
        # 不一定哪一个好，你自己试试
        # optimization level of layerwise equalization
        # 1 - single branch equalization(can not cross add, sub)
        # 2 - multi branch equalization(equalization cross add, sub)
        # don't know which one is better, try it by yourself.
        self.opt_level            = 1
        
        # Equalization 迭代次数，试试 1，2，3，10，100
        # algorithm iteration times, try 1, 2, 3, 10, 100
        self.iterations           = 10
        
        # Equalization 权重阈值，试试 0.5, 2
        # 这是个十分重要的属性，所有小于该值的权重不会参与运算
        # value threshold of equalization, try 0.5 and 2
        # it is a curical setting of equalization, value below this threshold won't get included in this optimizition.
        self.value_threshold      = .5 # try 0.5 and 2, it matters.
        
        # 是否在 Equalization 中拉平 bias
        # whether to equalize bias as well as weight
        self.including_bias       = False
        self.bias_multiplier      = 0.5
        
        # 是否在 Equalization 中拉平 activation
        # whether to equalize activation as well as weight
        self.including_act        = False
        self.act_multiplier       = 0.5

        # 暂时没用
        # for now this is a useless setting.
        self.self_check           = False


class ChannelSplitSetting():
    def __init__(self) -> None:
        # 所有需要分裂的层的名字，Channel Split 会降低网络执行的性能，你必须手动指定那些层要被分裂
        # interested layer names on which channel split is desired
        self.interested_layers = []
        # 分裂方向 - 可以是向上分裂或者向下分裂，每个层都要给一个哦
        # search direactions of layers in interested_layers, can be 'down' or 'up', each layer has one.
        self.search_directions = []
        # 要分裂多少 channel，数值越高则越多 channel 会被分裂
        # ratio of newly added channels
        self.expand_ratio      =  0.1
        # 还没看，我也不知道是什么
        # value split ratio 
        self.split_ratio       =  0.5
        # 是否添加一个小偏移项用来使得量化后的结果尽可能与浮点对齐。
        # cancel out round effect
        self.grid_aware        =  True
    

class MatrixFactorizationSetting():
    def __init__(self) -> None:
        # 所有需要分裂的层的名字，Matrix Factorization 会降低网络执行的性能，你必须手动指定那些层要被分裂
        # interested layer names on which channel split is desired
        self.interested_layers = []
        
        self.method = 'svd'


class AdvancedOptimizationSetting():
    def __init__(self) -> None:
        # 中间结果保存位置，可以选择 executor 或 cpu
        # executor - 将中间结果保存在 executor 的执行设备上(通常是cuda)
        # cpu - 将中间结果保存在 cpu memory 上
        # device to store all generated data.
        # executor - store data to executing device. (cuda)
        # cpu - store data to cpu memory.
        self.collecting_device    = 'executor' # executor or cpu
        
        # 每一轮迭代后是否校验优化结果，开启后延长执行时间，提升精度
        # whether to check optimization result after each iteration.
        self.auto_check           = False
        
        # 偏移量限制，试试 2, 4, 10
        # offset limitation used in this optimziation, try 2, 4, 10
        self.limit                = 4.0

        # 学习率
        # learning rate.
        self.lr                   = 1e-3
        
        # 训练步数
        # training steps
        self.steps                = 2500
        
        # 对那些层进行训练，默认为 空数组 则训练全部层
        # layers that need to be finetuned via this method.
        # Empty list means all layers need to be finetuned.
        self.interested_layers = []
        
        # 一个 variable 列表，指明需要针对那些 variable 进行优化
        # a list of variable(name), all variable listed will be
        # optimized via this pass. by default, all output variables
        # will be listed here.(if empty list was passed)
        self.interested_outputs = []
        
        self.verbose            = True


class ActivationQuantizationSetting():
    def __init__(self) -> None:
        # 激活值校准算法，不区分大小写，可以选择 minmax, kl, percentile, MSE
        # activation calibration method
        self.calib_algorithm = 'percentile'

        # 执行逐层激活值校准，延长执行时间，提升精度
        # whether to calibrate activation per - layer.
        self.per_layer_calibration = False

        # 激活值原地量化，设置为 True 且 USING_CUDA_KERNEL = True，则所有激活值原地量化，不产生额外显存
        # inplace quantization, if USING_CUDA_KERNEL = True, 
        # quantize all activations inplace, do not require extra memory.
        self.inplace_act_quantization = False


class ParameterQuantizationSetting():
    def __init__(self) -> None:
        # 参数校准算法，不区分大小写，可以选择 minmax, percentile(per-layer), kl(per-layer sym), MSE
        # parameter calibration method
        self.calib_algorithm = 'minmax'
        
        # 是否处理被动量化参数
        # whether to process passive parameters
        self.quantize_passive_parameter = True
        
        # 是否执行参数烘焙
        # whether to bake quantization on parameter.
        self.baking_parameter = True


class QuantizationFusionSetting():
    def __init__(self) -> None:
        # 一个你必须要启动的 Pass，修复所有算子上的定点错误
        # This pass is necessary and curicial, fix all quantization errors with your operation.
        self.refine_quantization = True
        
        # 一个你必须要启动的 Pass，删除无效定点。
        # PPQ 的定点过程与硬件设备一致，每一个算子都会尝试对它的输入和输出进行定点
        # 但在网络当中，当输入已经被上游算子进行了定点，则当前算子无需再对其输入进行定点
        # This pass is necessary and curicial, remove all unnecessary quantization from your graph.
        # PPQ executor quantizes each input tensor and output tensor of current operation.
        # however if input tensor has been quantized by upstream operations, then there is no need to quantize it again.
        self.remove_useless_quantization = True
        
        # conv - relu - add 联合定点
        # conv - relu - add joint quantization.
        # only gpu platform support this joint quantization optimization.
        self.fuse_conv_add = False
        
        # computing operation - activation operation 联合定点
        # joint quantization with computing operations and following activations.
        self.fuse_activation = True
        
        # 省略被动算子的定点，保持算子前后定点信息一致
        # skip passive opeartion's input and output quantization, keep them with a same quantization scale and offset.
        self.fuse_passive_op = True
        
        # 对多输入算子执行定点对齐
        # 对于 Add, Concat 这类具有多个输入的算子，硬件执行时必须要求所有输入位宽相同且量化信息一致
        # 在 8 bit 网络中，我们不讨论位宽的问题，只讨论量化信息
        # 选择对齐方式 - Align to Large, 则所有输入量化信息将合并成一个更大范围的量化信息, 
        #   保证其能表示所有输入的最小值与所有输入的最大值。
        # 选择对齐方式 - Align to Output, 则所有输入量化信息将全部等同于算子输出的量化信息
        # 设置 force_alignment_overlap 为 True, 则强制覆盖算子的上游量化信息，对于特殊结构的网络，这可能导致大范围的合并。
        
        # For Operations like Add, Concat, hardware requires their inputs share a same quantization config.
        # So that we implements 2 alignment method here for simulating hardware behaviour:
        # Align to Large: all input variables will merge their quantization range to a larger one.
        # Align to Output: all input quantization config will be replaced by output quantization config.
        # force_alignment_overlap: if set to true, ppq will overlap upstream quantization config, for some networks
        #   it will incurs configuration coalesce among the whole graph.
        self.align_quantization   = True
        self.align_elementwise_to = 'Align to Large'
        self.align_concat_to      = 'Align to Output'
        self.force_alignment_overlap = False


class TemplateSetting():
    """
    TemplateSetting 只是为了向你展示如何创建一个新的 Setting 并传递给相对应的 pass
        传递的过程在 ppq.quantization.quantizer.base.py 里面
   
    TemplateSetting just shows you how to create a setting class.
        parameter passing is inside ppq.quantization.quantizer.base.py

    Raises:
        TypeError: [description]

    Returns:
        [type]: [description]
    """
    def __init__(self) -> None:
        self.my_first_parameter = 'This Parameter just shows you how to create your own parameter.'


class BiasCorrectionSetting():
    def __init__(self) -> None:
        # 一个 variable 列表，指明需要针对那些 variable 进行优化
        # a list of variable(name), all variable listed will be
        # optimized via this pass. by default, all output variables
        # will be listed here.(if empty list was passed)
        self.interested_outputs = None
        
        # 每一轮迭代后是否校验优化结果，建议开启，延长执行时间，提升精度
        # whether to check optimization result after each iteration.
        self.auto_check         = True
        
        self.max_steps          = 8
        
        self.verbose            = True


class LearningStepSizeSetting():
    def __init__(self) -> None:

        # should only contain computing ops, if given, ops in interested_layers will be checked and
        # if conditions are satisfied, weight, scale of weight, scale of activation will be trained
        # with mse optimization goal, if not given, every condition-satisfied computing op will be 
        # optimized
        self.interested_layers      = []

        # if set True, only operations in interested_layers will be tuned, and if set False, ops in
        # the same subgraph(block) will be tuned, it's hard to say which one is better, you need to
        # try yourself
        self.interested_layers_only = False

        # num of training epochs, please adjust it to your needs
        self.epochs                 = 30

        # initial learning rate, by default Adam optimizer and a multistep scheduler with 0.1 decay
        # are used for convergence
        self.lr                     = 5e-5

        # scale multiplifer for bias(passive quantized param)
        self.scale_multiplier       = 2.0

        # global or local, if mode is set to global, you should make sure valid gradient could flow back
        # to your parameters from variable specified in output_names, by default the graph outputs will be
        # used for loss computing and gradient backward
        self.mode                   = 'global'

        # variable names to compute loss, if not given, the final output will be used
        # in graphwise mode, be careful in aware of valid back propagation in your graph
        self.output_names           = []

        # only useful when mode is global, should be a dict specifying how much each output weighes when 
        # multiple output names are given in graphwise mode, by default every output will weigh equally to 
        # 1.0 when computing loss, but if you care some output more, you can make it weigh more by specifying
        # some larger value in loss_weights, i.e., self.loss_weights = {some_output_1:2.0, some_output_2 : 5.0, ...}
        self.loss_weights           = {}


class BlockwiseReconstructionSetting():
     def __init__(self) -> None:
        # if given, only block containing op in interested_layers will be optimized, otherwise every
        # block in graph will be optimized
        self.interested_layers  = []
        # whether to tune activation scale
        self.tune_act_scale     = True
        # initial learning rate, by default Adam optimizer
        self.lr                 = 1e-3
        # number of training epochs, 300 epochs would be enough for int8 networks
        self.epochs             = 300
        # loss = LpNormLoss + lamda * RoundingLoss
        self.lamda              = 1.0
        # scale multiplifer for bias(passive quantized param)
        self.scale_multiplier   = 2.0



class DispatchingTable():
    def __init__(self) -> None:
        self.intro_0 = "Dispatching Table is a str -> TargetPlatform dictionary."
        self.intro_1 = "Any thing listed in this table will override the policy of PPQ dispatcher."
        self.intro_2 = "Example above shows you how to edit a valid dispatching table."
        self.attention = "All operation names that can not be found with your graph will be ignored."

        self.dispatchings = {
            'YOUR OEPRATION NAME' : 'TARGET PLATFORM(INT)',
            'FP32 OPERATION NAME' : TargetPlatform.FP32.value,
            'SOI OPERATION NAME'  : TargetPlatform.SHAPE_OR_INDEX.value,
            'DSP INT8 OPERATION NAME' : TargetPlatform.PPL_DSP_INT8.value,
            'TRT INT8 OPERATION NAME' : TargetPlatform.TRT_INT8.value,
            'NXP INT8 OPERATION NAME' : TargetPlatform.NXP_INT8.value,
            'PPL INT8 OPERATION NAME' : TargetPlatform.PPL_CUDA_INT8.value
        }
    
    def append(self, operation: str, platform: Union[int, TargetPlatform]):
        assert isinstance(platform, int) or isinstance(platform, TargetPlatform), (
            'Your dispatching table contains a invalid setting of operation {operation}, '
            'All platform setting given in dispatching table is expected given as int or TargetPlatform instance, '
            f'however {type(platform)} was given.'
        )
        if isinstance(platform, TargetPlatform): platform = platform.value
        self.dispatchings.update({operation: platform})


class QuantizationSetting():
    def __init__(self) -> None:
        # 子图切分与调度算法，可从 'aggresive', 'conservative', 'PPLNN' 中三选一，不区分大小写
        self.dispatcher                      = 'conservative'
        
        self.graph_format_setting            = GraphFormatSetting()
        
        # ssd with loss check equalization 相关设置
        # may take longer time(about 30min for default 3 iterations), but guarantee better result than baseline
        # should not be followed by a plain equalization when turned on
        self.ssd_equalization                = False
        self.ssd_setting                     = SSDEqualizationSetting()

        # layer wise equalizition 与相关配置
        self.equalization                    = False
        self.equalization_setting            = EqualizationSetting()
        
        # OCS channel split configuration
        self.channel_split                   = False
        self.channel_split_setting           = ChannelSplitSetting()
        
        # Matrix Factorization Split. (Experimental method)
        self.matrix_factorization            = False
        self.matrix_factorization_setting    = MatrixFactorizationSetting()

        # activation 量化与相关配置
        self.quantize_activation             = True
        self.quantize_activation_setting     = ActivationQuantizationSetting()

        # 参数量化与相关配置 
        self.quantize_parameter              = True
        self.quantize_parameter_setting      = ParameterQuantizationSetting()


        self.lsq_optimization                = False
        self.lsq_optimization_setting        = LearningStepSizeSetting()

        
        self.blockwise_reconstruction         = False
        self.blockwise_reconstruction_setting = BlockwiseReconstructionSetting()


        # 是否启动优化算法降低量化误差
        self.advanced_optimization           = False
        self.advanced_optimization_setting   = AdvancedOptimizationSetting()
        
        # 是否启动 bias correction pass
        self.bias_correct                    = False
        self.bias_correct_setting            = BiasCorrectionSetting()

        # 量化融合相关配置
        self.fusion                          = True
        self.fusion_setting                  = QuantizationFusionSetting()
        
        # extension setting 只是一个空白的占位符，用来向你展示如何创建一个自定义的 setting 并传递参数。
        # extension setting shows you how to create a setting and pass parameter to passes.
        # see ppq.quantization.quantizer.base.py
        self.extension                       = False
        self.extension_setting               = TemplateSetting()

        # 程序签名
        self.version                         = PPQ_VERSION
        self.signature                       = PPQ_NAME
        
        # 算子调度表，你可以编辑它来手动调度算子。
        self.dispatching_table               = DispatchingTable()

    def to_json(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__, 
                          sort_keys=True, indent=4, ensure_ascii=False)


class QuantizationSettingFactory:
    @ staticmethod
    def default_setting() -> QuantizationSetting:
        return QuantizationSetting()

    @staticmethod
    def academic_setting() -> QuantizationSetting:
        default_setting = QuantizationSetting()
        default_setting.fusion = False
        return default_setting

    @ staticmethod
    def pplcuda_setting() -> QuantizationSetting:
        default_setting = QuantizationSetting()
        default_setting.equalization = False
        default_setting.fusion_setting.fuse_conv_add = False
        return default_setting

    @ staticmethod
    def metax_pertensor_setting() -> QuantizationSetting:
        default_setting = QuantizationSetting()
        default_setting.equalization = True
        default_setting.fusion_setting.fuse_conv_add = False
        return default_setting

    @ staticmethod
    def dsp_setting() -> QuantizationSetting:
        default_setting = QuantizationSetting()
        default_setting.equalization = True
        default_setting.equalization_setting.opt_level = 1
        default_setting.equalization_setting.value_threshold = .5
        default_setting.equalization_setting.iterations = 3
        
        default_setting.fusion_setting.fuse_conv_add = False
        return default_setting
    
    @ staticmethod
    def nxp_setting() -> QuantizationSetting:
        default_setting = QuantizationSetting()
        default_setting.equalization = False
        default_setting.fusion_setting.fuse_conv_add = False
        return default_setting

    @ staticmethod
    def from_json(json_obj: Union[str, dict]) -> QuantizationSetting:
        setting = QuantizationSetting()
        if isinstance(json_obj, str):
            setting_dict = json.loads(json_obj)
        else: setting_dict = json_obj
        
        if 'version' not in setting_dict:
            ppq_warning('Can not find a valid version from your json input, input might not be correctly parsed.')
        else:
            version = setting_dict['version']
            if version < PPQ_VERSION:
                ppq_warning(f'You are loading a json quantization setting created by PPQ of another version: '
                            f'({version}), input setting might not be correctly parsed. '
                            'And all missing attributes will set as default without warning.')
        
        if not isinstance(setting_dict, dict):
            raise TypeError('Can not load setting from json string, '
                            f'expect a dictionary deserialzed from json here, '
                            f'however {type(setting_dict)} was given.')
        
        def assign(obj_setting: dict, obj: object):
            for key, value in obj_setting.items():
                if key in obj.__dict__:
                    if 'builtin' in obj.__dict__[key].__class__.__module__:
                        obj.__dict__[key] = copy.deepcopy(value)
                    else:
                        assert isinstance(value, dict)
                        assign(value, obj.__dict__[key])
                else:
                    ppq_warning(
                    f'Unexpected attribute ({key}) was found in {obj.__class__.__name__}. '
                    'This might because you are using a setting generated by PPQ with another version. '
                    'We will continue setting loading progress, while this attribute will be skipped.')

        assign(setting_dict, setting)
        return setting
