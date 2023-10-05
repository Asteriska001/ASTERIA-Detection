# interfaces/base.py

from abc import ABC, abstractmethod

class CodeAnalyzerInterface(ABC):

    @abstractmethod
    def get_ast(self, file_path: str) -> object:
        pass

    @abstractmethod
    def get_cfg(self, file_path: str) -> object:
        pass

    @abstractmethod
    def get_pg(self, code: str) -> object:
        pass
