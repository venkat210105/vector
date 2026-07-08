from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class Point:
    id: str
    vector: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vector = np.asarray(self.vector, dtype=np.float32)
