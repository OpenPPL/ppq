# This file defines export functions & class of PPQ.
from ppq.api.setting import *
from ppq.core import *
from ppq.executor import TorchExecutor
from ppq.IR import BaseGraph, Operation, Variable
from ppq.IR.quantize import QuantableOperation, QuantableVariable
from ppq.parser import dump_graph_to_file, load_graph
from ppq.quantization.analyise.graphwise import graph_similarity_analyse
from ppq.quantization.measure import (torch_cosine_similarity,
                                      torch_cosine_similarity_as_loss,
                                      torch_KL_divergence,
                                      torch_mean_square_error, torch_snr_error)
from ppq.quantization.optim import (AdaRoundPass, BiasCorrectionPass,
                                    InplaceQuantizationSettingPass,
                                    LayerwiseEqualizationPass,
                                    PassiveParameterQuantizePass,
                                    NxpInputRoundingRefinePass,
                                    NxpQuantizeFusionPass,
                                    NXPResizeModeChangePass,
                                    ParameterBakingPass, ParameterQuantizePass,
                                    QuantizationOptimizationPass,
                                    QuantizationOptimizationPipeline,
                                    QuantizeFusionPass, QuantizeReducePass,
                                    QuantizeRefinePass, RuntimeCalibrationPass,
                                    RuntimePerlayerCalibrationPass)
from ppq.quantization.qfunction import BaseQuantFunction
from ppq.quantization.qfunction.linear import TorchLinearQuantFunction
from ppq.quantization.quantizer import (BaseQuantizer, NXP_Quantizer,
                                        PPLCUDAQuantizer,
                                        PPL_DSP_Quantizer, TensorRTQuantizer)
from ppq.scheduler import (AggresiveDispatcher, ConservativeDispatcher,
                           GraphDispatcher, PPLNNDispatcher)
from ppq.utils.round import (ppq_numerical_round, ppq_round_to_power_of_2,
                             ppq_tensor_round)
from ppq.log import *