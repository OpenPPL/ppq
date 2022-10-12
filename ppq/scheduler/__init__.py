from .base import (GraphDispatcher, reverse_tracing_pattern,
                   value_tracing_pattern)
from .dispatchers import AggresiveDispatcher, ConservativeDispatcher, PPLNNDispatcher, PointDispatcher
from .allin import AllinDispatcher
from .perseus import Perseus
# Do not forget register your dispather here.

DISPATCHER_TABLE = {
    "conservative": ConservativeDispatcher,
    "pplnn": PPLNNDispatcher,
    "aggresive": AggresiveDispatcher,
    "pointwise": PointDispatcher,
    "allin": AllinDispatcher,
    'perseus': Perseus
}