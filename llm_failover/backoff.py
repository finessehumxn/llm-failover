"""Exponential backoff with full jitter.

delay(n) = uniform(0, min(max_delay, base_delay * 2**(n-1)))

Full jitter (rather than a fixed or "equal jitter" schedule) spreads clients
that failed at the same moment across the whole window, so a provider that
just recovered is not hit by every retry at once.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Backoff:
    base_delay: float = 0.25
    max_delay: float = 4.0
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        if self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("delays must be >= 0")

    def ceiling(self, retry_number: int) -> float:
        """Upper bound of the window before retry number ``retry_number`` (1-based)."""
        if retry_number < 1:
            raise ValueError("retry_number is 1-based")
        # Cap the exponent so huge retry counts cannot overflow.
        return min(self.max_delay, self.base_delay * (2 ** min(retry_number - 1, 32)))

    def delay(self, retry_number: int) -> float:
        return self.rng.uniform(0.0, self.ceiling(retry_number))
