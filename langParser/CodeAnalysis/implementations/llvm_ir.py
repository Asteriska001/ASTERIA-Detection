# implementations/cpp_impl.py

from CodeAnalysis.interfaces.base import CodeAnalyzerInterface

from antlr4 import CommonTokenStream, StdinStream, FileStream

from CodeAnalysis.utils.llvmir_utils.antlr.gen.LLVMIRLexer import LLVMIRLexer
from CodeAnalysis.utils.llvmir_utils.antlr.gen.LLVMIRParser import LLVMIRParser

import programl as pg
    
class LLVMIRAnalyzer(CodeAnalyzerInterface):

    def get_ast(self, file_path: str) -> object:
        #获取AST
        stream = FileStream(file_path, encoding="utf8")
        lexer = LLVMIRLexer(stream)
        token_stream = CommonTokenStream(lexer)
        parser = LLVMIRParser(token_stream)
        parse_tree = parser.compilationUnit()
        #to process the tree
        #parse_tree.toStringTree()
        return parse_tree

    def get_cfg(self, file_path: str) -> object:
        # 实现CFG分析
        pass

        
    def get_pg(self, code: str) -> object:
        # 实现PDG分析
        G = pg.from_llvm_ir(code)
        dot_output = pg.to_dot(G)
        return dot_output
