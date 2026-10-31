"""Device kernels are imported lazily; compiler and CPU oracle need no NPU."""

from ..utils import get_tla

get_tla()
