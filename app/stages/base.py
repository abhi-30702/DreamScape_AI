from abc import ABC, abstractmethod
from pathlib import Path


class BaseStage(ABC):
    stage_num: int

    def __init__(self, cache_dir: Path, stub_stages: set[int]):
        self.cache_dir = cache_dir
        self.stub_mode = self.stage_num in stub_stages

    @abstractmethod
    def _run_real(self, input: dict) -> dict: ...

    @abstractmethod
    def _run_stub(self, input: dict) -> dict: ...

    def run(self, input: dict) -> dict:
        return self._run_stub(input) if self.stub_mode else self._run_real(input)
