# implementations/python_impl.py

import ast
import inspect
import os

from CodeAnalysis.interfaces.base import CodeAnalyzerInterface

from python_graphs import control_flow
from python_graphs import control_flow_graphviz
from python_graphs import program_utils

class PythonAnalyzer(CodeAnalyzerInterface):

    def get_ast(self, file_path: str) -> object:
        # 使用Python的AST库或其他工具实现
        with open(file_path, 'r') as file:
            source_code = file.read()
        
        tree = ast.parse(source_code)
        return tree

    def get_cfg(self, file_path: str) -> None:
        # 实现Python的CFG分析
        module = self._load_module_from_file(file_path)
        os.makedirs('out', exist_ok=True)

        # For each function in the loaded module, visualize its control flow graph.
        for name, fn in inspect.getmembers(module, predicate=inspect.isfunction):
            path = f'out/{name}_cfg.png'
            self._plot_control_flow_graph(fn, path)
        print('Done. See the `out` directory for the results.')

    def _load_module_from_file(self, file_path: str) -> object:
        import importlib.util
        spec = importlib.util.spec_from_file_location("module.name", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _plot_control_flow_graph(self, fn, path):
        graph = control_flow.get_control_flow_graph(fn)
        source = program_utils.getsource(fn)
        control_flow_graphviz.render(graph, include_src=source, path=path)

    def get_pg(self, file_path: str) -> None:
        # 实现Python的PDG分析
        module = self._load_module_from_file(file_path)
        os.makedirs('out', exist_ok=True)

        # For each function in the loaded module, visualize its program graph.
        for name, fn in inspect.getmembers(module, predicate=inspect.isfunction):
            path = f'out/{name}-program-graph.png'
            graph = program_graph.get_program_graph(fn)
            program_graph_graphviz.render(graph, path=path)
