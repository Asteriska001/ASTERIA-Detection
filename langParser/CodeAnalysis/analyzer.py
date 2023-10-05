# analyzer.py

from CodeAnalysis.implementations import python_impl, java_impl
from CodeAnalysis.implementations import c_impl, cpp_impl
from CodeAnalysis.implementations import llvm_ir_impl

class CodeAnalyzerFactory:

    @staticmethod
    def create_analyzer(language: str):
        if language == "python":
            return python_impl.PythonAnalyzer()
        elif language == "java":
            return java_impl.JavaAnalyzer()
        elif language == "c":
            return c_impl.CAnalyzer()
        elif language == "cpp":
            return cpp_impl.CppAnalyzer()
        elif language == "llvmir":
            return llvm_impl.LLVMIRAnalyzer()
        # ... 其他语言
        else:
            raise ValueError(f"No analyzer available for language: {language}")
