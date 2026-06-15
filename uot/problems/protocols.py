from typing import Protocol, runtime_checkable


@runtime_checkable
class HasExactCost(Protocol):
    def get_exact_cost(self) -> float: ...
