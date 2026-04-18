__all__ = [
    "FloatCallableLike",
    "IntCallableLike",
]

from collections.abc import Callable
from typing import Literal

from pylops.utils.typing import NDArray

FloatCallableLike = float | NDArray | Callable[[int], float | NDArray]
IntCallableLike = int | Callable[[int], int]

Tmemunit = Literal["B", "KB", "MB", "GB"]
