from typing import Dict, Callable

class Registry:
    def __init__(self): self._fns: Dict[str, Callable] = {}
    def register(self, name): 
        def deco(fn): self._fns[name] = fn; return fn
        return deco
    def get(self, name): return self._fns[name]

BACKBONES = Registry()
HEADS     = Registry()
DATASETS  = Registry()