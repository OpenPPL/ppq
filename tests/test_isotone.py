import torch

from ppq.IR import Variable
from ppq.lib import LinearQuantizationConfig
from ppq.quantization.observer import TorchIsotoneObserver, TorchMinMaxObserver
from ppq.quantization.qfunction import PPQLinearQuantFunction

from ppq import QuantizationStates

TQC = LinearQuantizationConfig()
v = Variable(name='TestVariable')

for i in range(10000):
    isotone_observer = TorchIsotoneObserver(v, TQC)
    minmax_observer  = TorchMinMaxObserver(v, TQC)

    TQC.state = QuantizationStates.INITIAL
    value = torch.softmax(torch.rand(size=[1, 10]), dim=-1)
    value, _ = torch.sort(value, dim=-1)
    isotone_observer.observe(value)
    isotone_observer.render_quantization_config()

    isotone_quant = PPQLinearQuantFunction(value, TQC)
    o = torch.argmax(value, dim=-1)
    q = torch.argmax(isotone_quant, dim=-1)
    isotone_error_num = torch.sum(o != q).item()
    isotone_scale     = TQC.scale

    TQC.state = QuantizationStates.INITIAL
    minmax_observer.observe(value)
    minmax_observer.render_quantization_config()

    minmax_quant = PPQLinearQuantFunction(value, TQC)
    o = torch.argmax(value, dim=-1)
    q = torch.argmax(minmax_quant, dim=-1)
    minmax_error_num = torch.sum(o != q).item()
    minmax_scale     = TQC.scale
    
    if not isotone_error_num <= minmax_error_num:
        print(isotone_observer.s_candidates)
        print(isotone_error_num, minmax_error_num)
        print(value)
        print(isotone_quant, isotone_scale)
        print(minmax_quant, minmax_scale)
        raise Exception('Test Failed.')


TQC = LinearQuantizationConfig(symmetrical=False)
v = Variable(name='TestVariable')

for i in range(10000):
    isotone_observer = TorchIsotoneObserver(v, TQC)
    minmax_observer  = TorchMinMaxObserver(v, TQC)

    TQC.state = QuantizationStates.INITIAL
    value = torch.softmax(torch.rand(size=[1, 10]), dim=-1)
    value, _ = torch.sort(value, dim=-1)
    isotone_observer.observe(value)
    isotone_observer.render_quantization_config()

    isotone_quant = PPQLinearQuantFunction(value, TQC)
    o = torch.argmax(value, dim=-1)
    q = torch.argmax(isotone_quant, dim=-1)
    isotone_error_num = torch.sum(o != q).item()
    isotone_scale     = TQC.scale

    TQC.state = QuantizationStates.INITIAL
    minmax_observer.observe(value)
    minmax_observer.render_quantization_config()

    minmax_quant = PPQLinearQuantFunction(value, TQC)
    o = torch.argmax(value, dim=-1)
    q = torch.argmax(minmax_quant, dim=-1)
    minmax_error_num = torch.sum(o != q).item()
    minmax_scale     = TQC.scale
    
    if not isotone_error_num <= minmax_error_num:
        print(isotone_observer.s_candidates)
        print(isotone_error_num, minmax_error_num)
        print(value)
        print(isotone_quant, isotone_scale)
        print(minmax_quant, minmax_scale)
        raise Exception('Test Failed.')