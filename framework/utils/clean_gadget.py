import re

# Keywords up to C11 and C++17; immutable set
CPP_KEYWORDS = frozenset({
    '__asm', '__builtin', '__cdecl', '__declspec', '__except', '__export',
    '__far16', '__far32', '__fastcall', '__finally', '__import', '__inline',
    '__int16', '__int32', '__int64', '__int8', '__leave', '__optlink',
    '__packed', '__pascal', '__stdcall', '__system', '__thread', '__try',
    '__unaligned', '_asm', '_Builtin', '_Cdecl', '_declspec', '_except',
    '_Export', '_Far16', '_Far32', '_Fastcall', '_finally', '_Import',
    '_inline', '_int16', '_int32', '_int64', '_int8', '_leave', '_Optlink',
    '_Packed', '_Pascal', '_stdcall', '_System', '_try', 'alignas', 'alignof',
    'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor', 'bool', 'break', 'case',
    'catch', 'char', 'char16_t', 'char32_t', 'class', 'compl', 'const',
    'const_cast', 'constexpr', 'continue', 'decltype', 'default', 'delete',
    'do', 'double', 'dynamic_cast', 'else', 'enum', 'explicit', 'export',
    'extern', 'false', 'final', 'float', 'for', 'friend', 'goto', 'if',
    'inline', 'int', 'long', 'mutable', 'namespace', 'new', 'noexcept', 'not',
    'not_eq', 'nullptr', 'operator', 'or', 'or_eq', 'override', 'private',
    'protected', 'public', 'register', 'reinterpret_cast', 'return', 'short',
    'signed', 'sizeof', 'static', 'static_assert', 'static_cast', 'struct',
    'switch', 'template', 'this', 'thread_local', 'throw', 'true', 'try',
    'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using', 'virtual',
    'void', 'volatile', 'wchar_t', 'while', 'xor', 'xor_eq', 'NULL'
})

""" 
with open(r"/home/chengxiao/project/vul_detect/resources/sensiAPI.txt", "r") as file:
    additional_keywords = file.read().split(',')
CPP_KEYWORDS = CPP_KEYWORDS.union(additional_keywords) 
"""

# Constants to identify non-user-defined functions and arguments in main function
MAIN_FUNCTIONS = frozenset({'main'})
MAIN_ARGUMENTS = frozenset({'argc', 'argv'})

def clean_gadget(code_block):
    # Maps function names to unique symbols
    function_map = {}
    # Maps variable names to unique symbols
    variable_map = {}

    function_counter = 1
    variable_counter = 1

    # Regular expressions
    multiline_comment_regex = re.compile('\*/\s*$')
    function_regex = re.compile(r'\b([_A-Za-z]\w*)\b(?=\s*\()')
    variable_regex = re.compile(r'\b([_A-Za-z]\w*)\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()')

    sanitized_code = []

    for line in code_block:
        if not multiline_comment_regex.search(line):
            # Remove string and character literals, and non-ASCII characters
            line = re.sub(r'".*?"', '""', line)
            line = re.sub(r"'.*?'", "''", line)
            line = re.sub(r'[^\x00-\x7f]', r'', line)

            # Find potential user-defined functions and variables
            functions_in_line = function_regex.findall(line)
            variables_in_line = variable_regex.findall(line)

            # Rename user-defined functions
            for func in functions_in_line:
                if func not in CPP_KEYWORDS and func not in MAIN_FUNCTIONS:
                    if func not in function_map:
                        function_map[func] = 'FUN' + str(function_counter)
                        function_counter += 1
                    line = re.sub(r'\b' + re.escape(func) + r'\b(?=\s*\()', function_map[func], line)

            # Rename user-defined variables
            for var in variables_in_line:
                if var not in CPP_KEYWORDS and var not in MAIN_ARGUMENTS:
                    if var not in variable_map:
                        variable_map[var] = 'VAR' + str(variable_counter)
                        variable_counter += 1
                    line = re.sub(r'\b' + re.escape(var) + r'\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()', variable_map[var], line)

            sanitized_code.append(line)

    return sanitized_code

if __name__ == '__main__':
    sample_code_1 = [
        '231 151712/shm_setup.c inputfunc 11',
        'int main(int argc, char **argv) {',
        'while ((c = getopt(argc, argv, "k:s:m:o:h")) != -1) {',
        'switch(c) {'
    ]

    sample_code_2 = [
        '278 151587/ffmpeg.c inputfunc 3159', 
        'int main(int argc,char **argv)',
        'ret = ffmpeg_parse_options(argc,argv);',
        'if (ret < 0) {'
    ]

    sample_code_3 = [
        'invalid_memory_access_012_s_001 *s;',
        's = (invalid_memory_access_012_s_001 *)calloc(1,sizeof(invalid_memory_access_012_s_001));',
        's->a = 20;', 's->b = 20;', 's->uninit = 20;', 'free(s);]'
    ]

    sample_gadgetline = [
        'function(File file, Buffer buff) /* this is a comment test */',
        'this is a comment test */'
    ]

    split_test = 'printf ( " " , variable ++  )'.split()

    print(clean_gadget(sample_code_1))
    print(clean_gadget(sample_code_2))
    print(clean_gadget(sample_code_3))
    print(clean_gadget(sample_gadgetline))
    print(split_test)
