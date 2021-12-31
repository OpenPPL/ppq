from ppq.IR import Operation
from ppq.IR import BaseGraph

class SimpleEqualizationPair:
    def __init__(self, input_conv: Operation, 
                 input_activation: Operation,
                 output_conv: Operation, 
                 output_activation: Operation) -> None:

        self.input_conv  = input_conv
        self.output_conv = output_conv
        self.input_activation  = input_activation
        self.output_activation = output_activation

        self.input_var   = input_conv.inputs[0]
        self.output_var  = output_conv.outputs[0]

        self.input_conv_range  = [None, None] # min, max range
        self.output_conv_range = [None, None] # min, max range
