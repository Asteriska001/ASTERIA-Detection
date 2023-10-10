# implementations/cpp_impl.py

from CodeAnalysis.interfaces.base import CodeAnalyzerInterface

from antlr4 import CommonTokenStream, StdinStream, FileStream

from CodeAnalysis.utils.c_utils.src.antlr.gen.CPP14_v2Lexer import CPP14_v2Lexer
from CodeAnalysis.utils.c_utils.src.antlr.gen.CPP14_v2Parser import CPP14_v2Parser
from CodeAnalysis.utils.c_utils.src.cfg_extractor.cfg_extractor_visitor import CFGExtractorVisitor
from CodeAnalysis.utils.c_utils.src.code_coverage.prime_path_coverage import prime_path_coverage_bruteforce, prime_path_coverage_superset
from CodeAnalysis.utils.c_utils.src.code_coverage.path_finder import prime_paths, simple_paths
from CodeAnalysis.utils.c_utils.src.graph.utils import last_node, head_node
from CodeAnalysis.utils.c_utils.src.graph.visual import draw_CFG

import programl as pg
    
class CppAnalyzer(CodeAnalyzerInterface):

    def get_ast(self, file_path: str) -> object:
        #获取AST
        stream = FileStream(file_path, encoding="utf8")
        lexer = CPP14_v2Lexer(stream)
        token_stream = CommonTokenStream(lexer)
        parser = CPP14_v2Parser(token_stream)
        parse_tree = parser.translationunit()
        #to process the tree
        #parse_tree.toStringTree()
        return parse_tree

    def get_cfg(self, file_path: str) -> object:
        # 实现CFG分析
        stream = FileStream(file_path, encoding="utf8")
        lexer = CPP14_v2Lexer(stream)
        token_stream = CommonTokenStream(lexer)
        parser = CPP14_v2Parser(token_stream)
        parse_tree = parser.translationunit()
        cfg_extractor = CFGExtractorVisitor()
        cfg_extractor.visit(parse_tree)
        funcs = cfg_extractor.functions
        for i, g in enumerate(funcs.values()):
            draw_CFG(g, f"../test_output/temp{i}", token_stream, verbose=False)

        
    def get_pg(self, code: str) -> object:
        # 实现PDG分析
        G = pg.from_cpp(code)
        dot_output = pg.to_dot(G)
        return dot_output
