"""
    PPQ Core Decorator & MetaClass definations
    PPQ 核心装饰器、元类型定义

    You are not allowed to modify this
    请勿修改此文件
"""

import gc
from typing import Callable
from torch.cuda import empty_cache

from .config import PPQ_DEBUG


class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
        possible methods include: base class, decorator, metaclass. We will use the
        metaclass because it is best suited for this purpose.

    see aslo: https://refactoring.guru/design-patterns/singleton/python/example
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


def ppq_legacy(version: str, adapt_to: str = None):
    def wrapper(func: Callable):
        if adapt_to is not None:
            return
            print(f'\033[31mfunction {func.__name__} is obsoleted since version {version}, '
                f'use {adapt_to} function instead.\033[0m')
        else:
            return
            print(f'\033[31mfunction {func.__name__} is obsoleted since version {version}, '
                f'contact with system developer to get more information.\033[0m')
        def _wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return _wrapper
    return wrapper


def empty_ppq_cache(func: Callable):
    """
        Using empty_ppq_cache decorator to clear ppq memory cache,
            both gpu memory and cpu memory will be clear via this function.
        Function which get decorated by this will clear all ppq system cache BEFORE its running.
    Args:
        func (Callable): decorated function
    """
    def _wrapper(*args, **kwargs):
        empty_cache() # torch.cuda.empty_cache might requires a sync of all cuda device.
        gc.collect()  # empty memory.
        return func(*args, **kwargs)
    return _wrapper


def ppq_quant_param_computing_function(func: Callable):
    """
        mark a function to be a scale-computing function.
    Args:
        func (Callable): decorated function
    """
    def _wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return _wrapper


def ppq_debug_function(func: Callable):
    """
        mark a function to be a debug function.
    Args:
        func (Callable): decorated function
    """
    def _wrapper(*args, **kwargs):
        if PPQ_DEBUG:
            debug_str = func(*args, **kwargs)
            if debug_str is None: return None
            assert isinstance(debug_str, str), (
                'ppq_debug_function should only return string instance, '
                f'while {str(func)} returns {type(debug_str)}')
            print(debug_str, end='')
        else: return None

    return _wrapper


def ppq_file_io(func: Callable):
    """
        mark a function to be a ppq file io function.
            function must have return a file handle.
    Args:
        func (Callable): decorated function
    """
    def _wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return _wrapper


def ppq_warning(info: str):
    print(f'\033[31m{info}\033[0m')


def ppq_info(info: str):
    print(f'\033[33m{info}\033[0m')