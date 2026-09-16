"""Local LIF dynamics over measured MaleCNS weights prepared by fly.ai.

This implementation is our model, not a biological recording. Range observations
are injected into looming feature neurons; photoreceptor vision is not simulated.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
import hashlib
import json
import math

from .world import Action, clip

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "male-cns"


def data_status(path=DATA):
    path = Path(path)
    ready = all((path / f).is_file() for f in ("weights.npz", "brain.npz"))
    return {"available": ready, "path": str(path),
            "source": "MaleCNS v1.0 / fly.ai processed connectivity",
            "message": "连接数据已就绪" if ready else "尚未下载连接数据，果蝇控制暂不可用"}


class FlyBrain:
    dt = 0.02

    def __init__(self, path=DATA, wiring="real", backend="scipy"):
        import numpy as np
        from scipy import sparse
        self.np = np
        path = Path(path)
        if not data_status(path)["available"]:
            raise FileNotFoundError("缺少 weights.npz / brain.npz，请先运行数据准备脚本")
        self.weights = sparse.load_npz(path / "weights.npz").tocsr().astype(np.float32)
        meta = np.load(path / "brain.npz", allow_pickle=False)
        self.types, self.sides = meta["cell_type"], meta["side"]
        self.n = self.weights.shape[0]
        if self.weights.shape != (self.n, self.n) or len(self.types) != self.n:
            raise ValueError("Weight and annotation dimensions do not agree")
        if not np.isfinite(self.weights.data).all():
            raise ValueError("Weights contain non-finite entries")
        self.groups = {}
        for side in ("L", "R"):
            self.groups["input_"+side] = np.flatnonzero(np.isin(self.types, ["LC4", "LPLC2"]) & (self.sides == side))
            self.groups["output_"+side] = np.flatnonzero((self.types == "DNp01") & (self.sides == side))
        if any(len(g) == 0 for g in self.groups.values()):
            raise ValueError("Required left/right looming and descending groups are missing")
        if wiring == "shuffled":
            # Relabel destinations only. Preserve edge values and outgoing degrees;
            # not a degree-preserving swap control (explicitly named in results).
            perm = np.random.default_rng(2026).permutation(self.n)
            self.weights = self.weights[perm].tocsr()
        elif wiring != "real":
            raise ValueError("Unknown wiring")
        self.wiring = wiring
        if backend not in ("scipy", "active"):
            raise ValueError("Unknown propagation backend")
        self.backend = backend
        if backend == "active":
            from .propagation import active_current
            self._columns = self.weights.tocsc()
            self._propagate = active_current
        self.reset(1)

    def reset(self, seed):
        np = self.np
        self.rng = np.random.default_rng(seed)
        self.voltage = np.zeros(self.n, dtype=np.float32)
        self.spikes = np.zeros(self.n, dtype=np.float32)
        self.traces = np.zeros(2, dtype=np.float32)

    def act(self, observation):
        np = self.np
        start = perf_counter()
        p = [max(0, 1-r/2.5) for r in observation["rays"]]
        left, right = max(p[4:]), max(p[:5])
        # A small fixed input asymmetry breaks exact frontal symmetry. This is
        # an encoder choice, not evidence of an innate turning preference.
        drives = [0.85*left, 0.90*right]
        total = 0
        for _ in range(5):
            self.voltage *= math.exp(-self.dt/0.1)
            if self.backend == "active":
                c = self._columns
                current = self._propagate(c.indptr, c.indices, c.data, self.spikes)
            else:
                current = self.weights @ self.spikes
            self.voltage += 3.0*current + 0.14
            self.voltage += (self.rng.random(self.n) < 1.2*self.dt)*0.22
            for i, side in enumerate(("L", "R")):
                self.voltage[self.groups["input_"+side]] += drives[i]
            self.spikes = (self.voltage >= 1.0).astype(np.float32)
            self.voltage[self.spikes > 0] = 0
            total += int(self.spikes.sum())
            rates = np.array([self.spikes[self.groups["output_"+s]].mean()/self.dt for s in ("L", "R")])
            self.traces = 0.82*self.traces + 0.18*rates
        lrate, rrate = map(float, self.traces)
        # The sensor-to-feature and neuron-to-action mappings are engineered.
        residual = Action(-0.65*clip((lrate+rrate)/50, 0, 1),
                          clip((rrate-lrate)*0.09, -1.8, 1.8))
        return residual, {"left_hz": lrate, "right_hz": rrate,
                          "left_input": left, "right_input": right,
                          "spikes": total, "neurons": self.n, "edges": self.weights.nnz,
                          "compute_ms": round((perf_counter()-start)*1000, 1),
                          "wiring": self.wiring, "learning": False, "backend": self.backend}
