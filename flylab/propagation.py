"""CPU propagation visiting only presynaptic neurons that actually fired."""
import numpy as np
from numba import njit


@njit(cache=True)
def active_current(indptr, indices, weights, spikes):
    # CSC columns are traversed in ascending neuron order, matching the order
    # of nonzero contributions in sorted CSR rows. No fast-math or parallel sum.
    current = np.zeros(len(spikes), dtype=np.float32)
    for pre in range(len(spikes)):
        if spikes[pre] != 0:
            for edge in range(indptr[pre], indptr[pre+1]):
                current[indices[edge]] += weights[edge]*spikes[pre]
    return current
