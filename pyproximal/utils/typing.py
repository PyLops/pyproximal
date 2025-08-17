__all__ = [
    "FloatCallableLike",
    "IntCallableLike",
]

from typing import Callable, Union
from pylops.utils.typing import NDArray


FloatCallableLike = Union[float, NDArray, Callable[[int], Union[float, NDArray]]]
IntCallableLike = Union[int, Callable[[int], int]]

