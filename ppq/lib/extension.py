from ppq.core import NetworkFramework, TargetPlatform
from ppq.executor import register_operation_handler
from ppq.IR import GraphExporter, GraphBuilder
from ppq.quantization.observer import OBSERVER_TABLE, OperationObserver
from ppq.quantization.quantizer import BaseQuantizer

from .common import __EXPORTERS__, __PARSERS__, __QUANTIZER_COLLECTION__


def register_network_quantizer(quantizer: type, platform: TargetPlatform):
    """Register a quantizer to ppq quantizer collection.
    
    This function will override the default quantizer collection:
        register_network_quantizer(MyQuantizer, TargetPlatform.TRT_INT8) will replace the default TRT_INT8 quantizer.

    Quantizer should be a subclass of BaseQuantizer, do not provide an instance here as ppq will initilize it later.
    Your quantizer must require no initializing params.

    Args:
        quantizer (type): quantizer to be inserted.
        platform (TargetPlatform): corresponding platfrom of your quantizer.
    """
    if not isinstance(quantizer, type):
        raise TypeError(f'You can only register a class type as custimized ppq quantizer, '
                        f'however {type(quantizer)} is given. '
                        '(Requiring a class type here, do not provide an instance)')
    if not issubclass(quantizer, BaseQuantizer):
        raise TypeError('You can only register a subclass of BaseQuantizer as custimized quantizer.')
    __QUANTIZER_COLLECTION__[platform] = quantizer


def register_network_parser(parser: type, framework: NetworkFramework):
    """Register a parser to ppq parser collection. 

    This function will override the default parser collection:
        register_network_parser(MyParser, NetworkFramework.ONNX) will replace the default ONNX parser.

    Parser should be a subclass of GraphBuilder, do not provide an instance here as ppq will initilize it later.
    Your quantizer must require no initializing params.

    Args:
        parser (type): parser to be inserted.
        framework (NetworkFramework): corresponding NetworkFramework of your parser.
    """
    if not isinstance(parser, type):
        raise TypeError(f'You can only register a class type as custimized ppq parser, '
                        f'however {type(parser)} is given. '
                        f'(Requiring a class type here, do not provide an instance)')
    if not issubclass(parser, GraphBuilder):
        raise TypeError('You can only register a subclass of GraphBuilder as custimized parser.')
    __PARSERS__[framework] = parser


def register_network_exporter(exporter: type, platform: TargetPlatform):
    """Register an exporter to ppq exporter collection.

    This function will override the default exporter collection:
        register_network_quantizer(MyExporter, TargetPlatform.TRT_INT8) will replace the default TRT_INT8 exporter.

    Exporter should be a subclass of GraphExporter, do not provide an instance here as ppq will initilize it later.
    Your Exporter must require no initializing params.

    Args:
        exporter (type): exporter to be inserted.
        platform (TargetPlatform): corresponding platfrom of your exporter.
    """
    if not isinstance(exporter, type):
        raise TypeError(f'You can only register a class type as custimized ppq exporter, '
                f'however {type(exporter)} is given. '
                f'(Requiring a class type here, do not provide an instance)')
    if not issubclass(exporter, GraphExporter):
        raise TypeError('You can only register a subclass of GraphExporter as custimized exporter.')
    __EXPORTERS__[platform] = exporter


def register_calibration_observer(algorithm: str, observer: type):
    """Register an calibration observer to OBSERVER_TABLE.

    This function will override the existing OBSERVER_TABLE without warning.
    
    registed observer must be a sub class of OperationObserver.

    Args:
        exporter (type): exporter to be inserted.
        platform (TargetPlatform): corresponding platfrom of your exporter.
    """
    if not isinstance(observer, type):
        raise TypeError(
            f'You can only register an OperationObserver as custimized ppq observer, '
            f'however {type(observer)} is given. ')
    if not issubclass(observer, OperationObserver):
        raise TypeError('Regitsing observer must be a subclass of OperationObserver.')
    OBSERVER_TABLE[algorithm] = observer


__all__ = ['register_network_quantizer', 'register_network_parser', 
    'register_network_exporter', 'register_operation_handler', 
    'register_calibration_observer']
