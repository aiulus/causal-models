import numpy as np
from typing import Dict, List, Callable

from discarded.scm import SCM


class Sampler:
    def __init__(self, scm: SCM):
        self.scm = scm

    def sample_L1(self, n_samples: int) -> Dict[str, np.ndarray]:
        return self.scm.sample(n_samples)

    def sample_L2(self, n_samples: int, interventions: List[Callable[[SCM], None]]) -> Dict[str, np.ndarray]:
        for fn in interventions:
            self.scm.apply_intervention(fn)
        return self._sample(n_samples)

    def sample_L3(self, n_samples: int, observations: List[Callable[[SCM], None]],
                  interventions: List[Callable[[SCM], None]]) -> Dict[str, np.ndarray]:
        self.scm.apply_abduction(observations)
        for fn in interventions:
            self.scm.apply_intervention(fn)
        return self._sample(n_samples)
