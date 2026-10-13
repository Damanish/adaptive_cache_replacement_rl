"""
Cache simulators.

* MultiPolicyCache - a single physical cache that keeps the metadata needed by
  LRU, LFU and FIFO *simultaneously* (last access time, in-cache frequency,
  insertion time). The victim on a miss is chosen by whichever policy is active
  at that moment, so the RL agent can switch policy at any time without
  flushing or rebuilding the cache.
* ARCCache - Adaptive Replacement Cache (Megiddo & Modha, FAST'03), used as the
  classical *adaptive* baseline.
* belady_hits - Belady's MIN (clairvoyant optimum), used as an upper bound.
"""
from __future__ import annotations

import heapq
from collections import OrderedDict

import numpy as np

LRU, LFU, FIFO = 0, 1, 2
POLICY_NAMES = ("LRU", "LFU", "FIFO")


class MultiPolicyCache:
    __slots__ = ("C", "slot_of", "keys", "last", "freq", "ins", "size", "t")

    def __init__(self, capacity: int):
        self.C = capacity
        self.reset()

    def reset(self):
        C = self.C
        self.slot_of = {}
        self.keys = [-1] * C
        self.last = np.zeros(C, dtype=np.int64)
        self.freq = np.zeros(C, dtype=np.int64)
        self.ins = np.zeros(C, dtype=np.int64)
        self.size = 0
        self.t = 0

    # -- victim selection ------------------------------------------------------
    def _victim(self, policy: int) -> int:
        if policy == LRU:
            return int(self.last.argmin())
        if policy == FIFO:
            return int(self.ins.argmin())
        # LFU, ties broken by least-recent access
        return int((self.freq * (1 << 40) + self.last).argmin())

    # -- main entry points -----------------------------------------------------
    def access(self, key: int, policy: int) -> bool:
        """Serve one request; returns True on a hit."""
        self.t += 1
        s = self.slot_of.get(key)
        if s is not None:
            self.last[s] = self.t
            self.freq[s] += 1
            return True
        if self.size < self.C:
            s = self.size
            self.size += 1
        else:
            s = self._victim(policy)
            del self.slot_of[self.keys[s]]
        self.keys[s] = key
        self.slot_of[key] = s
        self.last[s] = self.t
        self.freq[s] = 1
        self.ins[s] = self.t
        return False

    def run(self, keys, policy: int) -> int:
        """Serve a block of requests with a fixed policy; returns #hits."""
        hits = 0
        acc = self.access
        for k in keys:
            hits += acc(int(k), policy)
        return hits

    def frac_freq_one(self) -> float:
        """Fraction of resident blocks that were never re-referenced."""
        if self.size == 0:
            return 0.0
        return float((self.freq[: self.size] == 1).mean())


def fixed_policy_hits(keys: np.ndarray, capacity: int, policy: int) -> np.ndarray:
    """Per-request hit indicator (uint8) for a fixed classical policy."""
    c = MultiPolicyCache(capacity)
    acc = c.access
    out = np.empty(len(keys), dtype=np.uint8)
    for i, k in enumerate(keys.tolist()):
        out[i] = acc(k, policy)
    return out


# --------------------------------------------------------------------------- #
#  ARC
# --------------------------------------------------------------------------- #
class ARCCache:
    def __init__(self, capacity: int):
        self.c = capacity
        self.p = 0.0
        self.T1, self.T2 = OrderedDict(), OrderedDict()
        self.B1, self.B2 = OrderedDict(), OrderedDict()

    def _replace(self, in_b2: bool):
        if self.T1 and (len(self.T1) > self.p or (in_b2 and len(self.T1) == int(self.p))
                        or not self.T2):
            k, _ = self.T1.popitem(last=False)
            self.B1[k] = None
        else:
            k, _ = self.T2.popitem(last=False)
            self.B2[k] = None

    def access(self, x) -> bool:
        c = self.c
        if x in self.T1:
            del self.T1[x]
            self.T2[x] = None
            return True
        if x in self.T2:
            self.T2.move_to_end(x)
            return True
        if x in self.B1:
            self.p = min(c, self.p + max(len(self.B2) / len(self.B1), 1.0))
            self._replace(False)
            del self.B1[x]
            self.T2[x] = None
            return False
        if x in self.B2:
            self.p = max(0.0, self.p - max(len(self.B1) / len(self.B2), 1.0))
            self._replace(True)
            del self.B2[x]
            self.T2[x] = None
            return False
        # complete miss
        l1 = len(self.T1) + len(self.B1)
        total = l1 + len(self.T2) + len(self.B2)
        if l1 == c:
            if len(self.T1) < c:
                self.B1.popitem(last=False)
                self._replace(False)
            else:
                self.T1.popitem(last=False)
        elif total >= c:
            if total >= 2 * c:
                self.B2.popitem(last=False)
            self._replace(False)
        self.T1[x] = None
        return False


def arc_hits(keys: np.ndarray, capacity: int) -> np.ndarray:
    a = ARCCache(capacity)
    out = np.empty(len(keys), dtype=np.uint8)
    for i, k in enumerate(keys.tolist()):
        out[i] = a.access(k)
    return out


# --------------------------------------------------------------------------- #
#  Belady MIN (offline optimum)
# --------------------------------------------------------------------------- #
def belady_hits(keys: np.ndarray, capacity: int) -> np.ndarray:
    keys = keys.tolist()
    n = len(keys)
    nxt = [n + 1] * n
    seen = {}
    for i in range(n - 1, -1, -1):
        nxt[i] = seen.get(keys[i], n + i + 1)  # unique "never again" stamps
        seen[keys[i]] = i
    cache = {}           # key -> next use
    heap = []            # (-next_use, key), lazy deletion
    out = np.zeros(n, dtype=np.uint8)
    for i, k in enumerate(keys):
        if k in cache:
            out[i] = 1
        elif len(cache) >= capacity:
            while True:
                nu, v = heapq.heappop(heap)
                if cache.get(v) == -nu:
                    del cache[v]
                    break
        cache[k] = nxt[i]
        heapq.heappush(heap, (-nxt[i], k))
    return out
