# implementations/java_impl.py

from CodeAnalysis.interfaces.base import CodeAnalyzerInterface

from antlr4 import CommonTokenStream, StdinStream, FileStream

from CodeAnalysis.utils.java_utils.antlr.JavaLexer import JavaLexer
from CodeAnalysis.utils.java_utils.antlr.JavaParser import JavaParser

from program_graphs.cfg import parse_java as parse_java_cfg
from program_graphs.adg import parse_java as parse_java_adg

class JavaAnalyzer(CodeAnalyzerInterface):

    def get_ast(self, file_path: str) -> object:
        #获取AST
        stream = FileStream(file_path, encoding="utf8")
        lexer = JavaLexer(stream)
        token_stream = CommonTokenStream(lexer)
        parser = JavaParser(token_stream)
        parse_tree = parser.compilationUnit()
        #to process the tree
        #parse_tree.toStringTree()
        return parse_tree

    def get_cfg(self, file_path: str) -> object:
        # 实现CFG分析
        with open(file_path, 'r') as f:
            code = f.read()
        cfg = parse_java_cfg(code)
        return cfg
        
    def get_adg(self, code: str) -> object:
        # 实现adg分析
        adg = parse_java_adg(code)
        return adg


if __name__ == '__main__':
    java_code = '''
    if (x  > 0) {
        y = 0;
    }
    '''

    graph = parse_java_adg(java_code)
    print(graph)