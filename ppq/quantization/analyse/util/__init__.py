from functools import partial
from typing import Dict
import math
import torch
from ppq.core.defs import ppq_warning
from ppq.quantization.measure.cosine import torch_cosine_similarity
from ppq.quantization.measure.norm import (torch_mean_square_error,
                                           torch_snr_error)


class MeasureRecorder():
    """Helper class for collecting data."""
    def __init__(self, measurement: str = 'cosine', reduce: str = 'mean') -> None:
        self.num_of_elements = 0
        self.measure         = 0
        if reduce not in {'mean', 'max'}:
            raise ValueError(f'PPQ MeasureRecorder Only support reduce by mean or max, however {reduce} was given.')

        if str(measurement).lower() == 'cosine':
            measure_fn = partial(torch_cosine_similarity, reduction=reduce)
        elif str(measurement).lower() == 'mse':
            measure_fn = partial(torch_mean_square_error, reduction=reduce)
        elif str(measurement).lower() == 'snr':
            measure_fn = partial(torch_snr_error, reduction=reduce)
        else:
            raise ValueError('Unsupported measurement detected. '
                f'PPQ only support mse, snr and consine now, while {measurement} was given.')

        self.measure_fn = measure_fn
        self.reduce     = reduce

    def update(self, y_pred: torch.Tensor, y_real: torch.Tensor):
        elements = y_pred.shape[0]
        if elements != y_real.shape[0]:
            raise Exception(
                'Can not update measurement, cause your input data do not share a same batchsize. '
                f'Shape of y_pred {y_pred.shape} - against shape of y_real {y_real.shape}')
        result = self.measure_fn(y_pred=y_pred, y_real=y_real).item()

        if self.reduce == 'mean':
            self.measure = self.measure * self.num_of_elements + result * elements
            self.num_of_elements += elements
            self.measure /= self.num_of_elements

        if self.reduce == 'max':
            self.measure = max(self.measure, result)
            self.num_of_elements += elements

class MeasurePrinter():
    """Helper class for print top-k record."""
    def __init__(self, data: Dict[str, float], measure: str, label: str = 'Layer',
                 k: int = None, order: str = 'large_to_small') -> None:
        if order not in {'large_to_small', 'small_to_large', None}:
            raise ValueError('Parameter "order" can only be "large_to_small" or "small_to_large"')
        self.collection = [(name, value) for name, value in data.items()]
        if order is not None:
            self.collection = sorted(self.collection, key=lambda x: x[1])
            if order == 'large_to_small': self.collection = self.collection[::-1]
        if k is not None: self.collection = self.collection[:k]

        if order is None:
            sorted_collection = sorted(self.collection, key=lambda x: x[1])
            largest_element, smallest_element = sorted_collection[-1][1], sorted_collection[0][1]
        elif order == 'large_to_small':
            largest_element, smallest_element = self.collection[0][1], self.collection[-1][1]
        else: largest_element, smallest_element = self.collection[-1][1], self.collection[0][1]
        self.normalized_by = largest_element - smallest_element
        self.min = smallest_element

        max_name_length = len(label)
        for name, _ in self.collection:
            max_name_length = max(len(name), max_name_length)
        self.max_name_length = max_name_length
        self.measure_str = measure
        self.label = label

    def print(self, max_blocks: int = 20):
        print(f'{self.label}{" " * (self.max_name_length - len(self.label))}  | {self.measure_str} ')
        for name, value in self.collection:
            normalized_value = (value - self.min) / (self.normalized_by + 1e-7)
            if math.isnan(value):
                ppq_warning('MeasurePrinter found an NaN value in your data.')
                normalized_value = 0
            num_of_blocks = round(normalized_value * max_blocks)
            print(f'{name}:{" " * (self.max_name_length - len(name))} | '
                  f'{"█" * num_of_blocks}{" " * (max_blocks - num_of_blocks)} | '
                  f'{value:.6f}')
