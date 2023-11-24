from .comm import NCCLCommBuilder
from .comm import MPICommBuilder
from .builder import get_default_compute_capabilities, OpBuilder

__op_builders__ = [
    NCCLCommBuilder(),
    MPICommBuilder(),
]
ALL_OPS = {op.name: op for op in __op_builders__}