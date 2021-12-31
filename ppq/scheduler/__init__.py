from .base import (GraphDispatcher, reverse_tracing_pattern,
                   value_tracing_pattern)
from .dispatchers import AggresiveDispatcher, ConservativeDispatcher, PPLNNDispatcher

# Do not forget register your dispather here.
DISPATCHER_TABLE = {
    'conservative': ConservativeDispatcher,
    'pplnn': PPLNNDispatcher,
    'aggresive': AggresiveDispatcher
}