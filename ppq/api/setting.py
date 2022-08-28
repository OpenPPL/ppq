import json
import copy
from typing import Union
from ppq.core import *

# ----------------------------------
# Legacy Settings, Following Settings are not used in PPQ 0.6.5
class AdvancedOptimizationSetting():
    def __init__(self) -> None:
        self.collecting_device    = 'executor'
        self.auto_check           = False
        self.limit                = 4.0
        self.lr                   = 1e-3
        self.steps                = 2500
        self.interested_layers = []
        self.interested_outputs = []
        self.verbose            = True

class MatrixFactorizationSetting():
    def __init__(self) -> None:
        self.interested_layers = []
        self.method = 'svd'
# ----------------------------------

class BlockwiseReconstructionSetting():
    def __init__(self) -> None:
        # 训练那些层，不设置则训练全部层
        self.interested_layers  = []
        
        # scale 是否可以训练
        self.is_scale_trainable = False
        
        # 学习率
        self.lr                 = 1e-3
        
        # 学习步数
        self.steps              = 5000
        
        # 正则化参数
        self.gamma              = 1.0
        
        # 缓存设备
        self.collecting_device  = 'cuda'
        
        # 区块大小
        self.block_size         = 4


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


class ChannelSplitSetting():
    def __init__(self) -> None:
        # 所有需要分裂的层的名字，Channel Split 会降低网络执行的性能，你必须手动指定那些层要被分裂
        # Weight Split 和 Channel Split 都是一种以计算时间作为代价，提高量化精度的方法
        # 这些方法主要适用于 per-tensor 的量化方案
        # interested layer names on which channel split is desired
        self.interested_layers = []
        # 分裂方向 - 可以是向上分裂或者向下分裂，每个层都要给一个哦
        # search direactions of layers in interested_layers, can be 'down' or 'up', each layer has one.
        self.search_directions = []
        # 要分裂多少 channel，数值越高则越多 channel 会被分裂
        # ratio of newly added channels
        self.expand_ratio      =  0.1
        # https://arxiv.org/abs/1901.09504 扩充 channel，为结果不变，需要新旧 channel 数值减半。不要修改此参数
        # value split ratio
        self.split_ratio       =  0.5
        # 是否添加一个小偏移项用来使得量化后的结果尽可能与浮点对齐。
        # cancel out round effect
        self.grid_aware        =  True


class BiasCorrectionSetting():
    def __init__(self) -> None:
        # 指定所有需要执行 BiasCorrection 的层的名字，不写就是所有层全部进行 bias correction
        self.interested_layers      = []

        # 指定 BiasCorrection 的区块大小，越大越快，但也越不精准
        self.block_size             = 4

        # 指定 bias 的统计步数，越大越慢，越小越不精准
        self.steps                  = 32

        # 缓存数据放在哪
        self.collecting_device      = 'executor'


class WeightSplitSetting():
    def __init__(self) -> None:
        # 所有需要分裂的层的名字，Weight Split 会降低网络执行的性能，你必须手动指定那些层要被分裂
        # Weight Split 和 Channel Split 都是一种以计算时间作为代价，提高量化精度的方法
        # 这些方法主要适用于 per-tensor 的量化方案
        # computing layers which are intended to be splited
        self.interested_layers = []
        
        # 所有小于阈值的权重将被分裂
        self.value_threshold   = 2.0
        
        # 分裂方式，可以选 balance(平均分裂), random(随机分裂)
        self.method            = 'balance'


class GraphFormatSetting():
    def __init__(self) -> None:
        # 有一些平台不支持Constant Op，这个pass会尝试将 Constant Operation 的输入转变为 Parameter Variable
        # Some deploy platform does not support constant operation,
        # this pass will convert constant operation into parameter variable(if can be done).
        self.format_constant_op = True

        # 融合Conv和Batchnorm
        # Fuse Conv and Batchnorm Layer. This pass is necessary and crucial.
        self.fuse_conv_bn       = True

        # 将所有的parameter variable进行分裂，使得每个variable具有至多一个输出算子
        # Split all parameter variables, making all variable has at most 1 output operation.
        # This pass is necessary and crucial.
        self.format_paramters   = True

        # 一个你必须要启动的 Pass
        # This pass is necessary and crucial.
        self.format_cast        = True

        # 尝试从网络中删除所有与输出无关的算子和 variable
        # Remove all unnecessary operations and variables(which are not link to graph output) from current graph,
        # notice that some platform use unlinked variable as quantization parameters, do not set this as true if so.
        self.delete_isolate     = True


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


class ActivationQuantizationSetting():
    def __init__(self) -> None:
        # 激活值校准算法，不区分大小写，可以选择 minmax, kl, percentile, MSE, None
        # 选择 None 时，将由 quantizer 指定量化算法
        # activation calibration method
        self.calib_algorithm = None


class ParameterQuantizationSetting():
    def __init__(self) -> None:
        # 参数校准算法，不区分大小写，可以选择 minmax, percentile, kl, MSE, None
        # parameter calibration method
        self.calib_algorithm = None

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

        # computing operation - activation operation 联合定点
        # joint quantization with computing operations and following activations.
        self.fuse_activation = True

        # 省略被动算子的定点，保持算子前后定点信息一致
        # skip passive operation's input and output quantization, keep them with a same quantization scale and offset.
        self.fuse_passive_op = True

        # 对多输入算子执行定点对齐
        # 对于 Add, Concat 这类具有多个输入的算子，硬件执行时必须要求所有输入位宽相同且量化信息一致
        # 在 8 bit 网络中，我们不讨论位宽的问题，只讨论量化信息
        # 选择对齐方式 - Align to Large, 则所有输入量化信息将合并成一个更大范围的量化信息,
        #   保证其能表示所有输入的最小值与所有输入的最大值。
        # 选择对齐方式 - Align to Output, 则所有输入量化信息将全部等同于算子输出的量化信息
        # 选择对齐方式 - None, 则不处理对应算子的量化对齐
        # 设置 force_alignment_overlap 为 True, 则强制覆盖算子的上游量化信息，对于特殊结构的网络，这可能导致大范围的合并。

        # For Operations like Add, Concat, hardware requires their inputs share a same quantization config.
        # So that we implements 2 alignment method here for simulating hardware behaviour:
        # Align to Large: all input variables will merge their quantization range to a larger one.
        # Align to Output: all input quantization config will be replaced by output quantization config.
        # None: take no alignment here, input and output will have their own(independent) quantization config.
        # force_alignment_overlap: if set to true, ppq will overlap upstream quantization config, for some networks
        #   it will incurs configuration coalesce among the whole graph.
        self.align_quantization      = True
        self.align_avgpooling_to     = 'None'
        self.align_elementwise_to    = 'Align to Large'
        self.align_concat_to         = 'Align to Output'
        self.force_alignment_overlap = False


class LSQSetting():
    def __init__(self) -> None:

        # should only contain computing ops, if given, ops in interested_layers will be checked and
        # if conditions are satisfied, weight, scale of weight, scale of activation will be trained
        # with mse optimization goal, if not given, every condition-satisfied computing op will be
        # optimized
        self.interested_layers      = []

        # initial learning rate, by default Adam optimizer and a multistep scheduler with 0.1 decay
        # are used for convergence
        self.lr                     = 1e-5

        # collecting device for block input and output
        # turn this to cpu if CUDA OOM Error
        self.collecting_device      = 'cuda'
        
        # num of training steps, please adjust it to your needs
        self.steps                  = 500
        
        # is scale trainable
        self.is_scale_trainable     = True
        
        # regularization term
        self.gamma                  = 0.0
        
        # block size of graph cutting.
        self.block_size             = 4


class TemplateSetting():
    """TemplateSetting 只是为了向你展示如何创建一个新的 Setting 并传递给相对应的 pass 传递的过程在
    ppq.quantization.quantizer.base.py 里面.

    TemplateSetting just shows you how to create a setting class.
        parameter passing is inside ppq.quantization.quantizer.base.py

    Raises:
        TypeError: [description]

    Returns:
        [type]: [description]
    """
    def __init__(self) -> None:
        self.my_first_parameter = 'This Parameter just shows you how to create your own parameter.'


class DispatchingTable():
    def __init__(self) -> None:
        self.intro_0 = 'Dispatching Table is a str -> TargetPlatform dictionary.'
        self.intro_1 = 'Any thing listed in this table will override the policy of PPQ dispatcher.'
        self.intro_2 = 'Example above shows you how to edit a valid dispatching table.'
        self.attention = 'All operation names that can not be found with your graph will be ignored.'

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
        # 子图切分与调度算法，可从 'pointwise', 'conservative', 'pursus', 'allin', 'pplnn' 中五选一，不区分大小写
        self.dispatcher                      = 'pursus'

        self.graph_format_setting            = GraphFormatSetting()

        # ssd with loss check equalization 相关设置
        # may take longer time(about 30min for default 3 iterations), but guarantee better result than baseline
        # should not be followed by a plain equalization when turned on
        self.ssd_equalization                = False
        self.ssd_setting                     = SSDEqualizationSetting()

        # layer wise equalizition 与相关配置
        self.equalization                    = False
        self.equalization_setting            = EqualizationSetting()

        self.weight_split                    = False
        self.weight_split_setting            = WeightSplitSetting()

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

        # 是否执行网络微调
        self.lsq_optimization                = False
        self.lsq_optimization_setting        = LSQSetting()


        self.blockwise_reconstruction         = False
        self.blockwise_reconstruction_setting = BlockwiseReconstructionSetting()

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
        self.version                         = PPQ_CONFIG.VERSION
        self.signature                       = PPQ_CONFIG.NAME

        # Following setting will be removed.
        self.advanced_optimization            = False
        self.advanced_optimization_setting    = AdvancedOptimizationSetting()

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
    
    @staticmethod
    def ncnn_setting() -> QuantizationSetting:
        default_setting = QuantizationSetting()
        default_setting.fusion = False
        return default_setting

    @ staticmethod
    def pplcuda_setting() -> QuantizationSetting:
        default_setting = QuantizationSetting()
        default_setting.equalization = False
        return default_setting

    @ staticmethod
    def metax_pertensor_setting() -> QuantizationSetting:
        default_setting = QuantizationSetting()
        default_setting.equalization = True
        return default_setting

    @ staticmethod
    def dsp_setting() -> QuantizationSetting:
        default_setting = QuantizationSetting()
        default_setting.equalization = True
        default_setting.equalization_setting.opt_level = 1
        default_setting.equalization_setting.value_threshold = .5
        default_setting.equalization_setting.iterations = 3

        return default_setting

    @ staticmethod
    def nxp_setting() -> QuantizationSetting:
        default_setting = QuantizationSetting()
        default_setting.equalization = False
        return default_setting

    @ staticmethod
    def fpga_setting() -> QuantizationSetting:
        default_setting = QuantizationSetting()
        default_setting.equalization = True
        default_setting.equalization_setting.including_bias = True
        return default_setting

    @staticmethod
    def trt_setting() -> QuantizationSetting:
        default_setting = QuantizationSetting()
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
            if version < PPQ_CONFIG.VERSION:
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
