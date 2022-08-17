# This file defines export functions & class of PPQ.
from ppq.api.setting import (ActivationQuantizationSetting, DispatchingTable,
                             EqualizationSetting, GraphFormatSetting,
                             LSQSetting, ParameterQuantizationSetting,
                             QuantizationFusionSetting, QuantizationSetting,
                             QuantizationSettingFactory, TemplateSetting)
from ppq.core import *
from ppq.executor import (BaseGraphExecutor, TorchExecutor,
                          TorchQuantizeDelegator)
from ppq.IR import (BaseGraph, GraphBuilder, GraphCommand, GraphExporter,
                    GraphFormatter, Operation, QuantableGraph, SearchableGraph,
                    Variable)
from ppq.IR.deploy import RunnableGraph
from ppq.IR.quantize import QuantableOperation, QuantableVariable
from ppq.IR.search import SearchableGraph
from ppq.log import NaiveLogger
from ppq.quantization.analyse import (graphwise_error_analyse,
                                      layerwise_error_analyse,
                                      parameter_analyse, statistical_analyse,
                                      variable_analyse)
from ppq.quantization.measure import (torch_cosine_similarity,
                                      torch_cosine_similarity_as_loss,
                                      torch_KL_divergence,
                                      torch_mean_square_error, torch_snr_error)
from ppq.quantization.optim import (BiasCorrectionPass, GRUSplitPass,
                                    HorizontalLayerSplitPass,
                                    LayerwiseEqualizationPass,
                                    MetaxGemmSplitPass, MishFusionPass,
                                    NxpInputRoundingRefinePass,
                                    NxpQuantizeFusionPass,
                                    NXPResizeModeChangePass,
                                    ParameterBakingPass, ParameterQuantizePass,
                                    PassiveParameterQuantizePass,
                                    QuantizationOptimizationPass,
                                    QuantizationOptimizationPipeline,
                                    QuantizeFusionPass, QuantizeSimplifyPass,
                                    QuantizeRefinePass, RuntimeCalibrationPass,
                                    SwishFusionPass)
from ppq.quantization.qfunction import BaseQuantFunction
from ppq.quantization.qfunction.linear import (PPQLinearQuant_toInt,
                                               PPQLinearQuantFunction)
from ppq.quantization.quantizer import (BaseQuantizer, NXP_Quantizer,
                                        PPL_DSP_Quantizer, PPLCUDAQuantizer,
                                        TensorRTQuantizer)
from ppq.scheduler import (AggresiveDispatcher, ConservativeDispatcher,
                           GraphDispatcher, PPLNNDispatcher)
from ppq.scheduler.perseus import Perseus
from ppq.utils.round import (ppq_numerical_round, ppq_round_to_power_of_2,
                             ppq_tensor_round)
