"""距离计算计数器。

为什么要有这个：QPS 依赖机器性能，评审换台机器数字就变了，不可比。
「每次查询触发多少次距离计算」是纯算法层指标，与硬件、语言、编译器全部无关。
它是本项目最有说服力的一条指标，因此所有距离计算必须严格经过这里。
"""

from __future__ import annotations


class DistanceCounter:
    __slots__ = ("count",)

    def __init__(self, start: int = 0) -> None:
        self.count = int(start)

    def add(self, n: int) -> None:
        self.count += int(n)

    def reset(self) -> None:
        self.count = 0

    def __int__(self) -> int:
        return self.count

    def __repr__(self) -> str:  # pragma: no cover
        return f"DistanceCounter({self.count:,})"
