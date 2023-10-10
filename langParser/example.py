#import sys
#sys.path.append("")
from CodeAnalysis import analyzer

analyzer_instance = analyzer.CodeAnalyzerFactory.create_analyzer("python")
ast_tree = analyzer_instance.get_ast("print('Hello, World!')")
