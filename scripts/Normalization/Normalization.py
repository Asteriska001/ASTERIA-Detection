# coding=utf-8
import os
import re
import argparse
from clean_gadget import clean_gadget

def parse_options():
    parser = argparse.ArgumentParser(description='Normalization.')
    parser.add_argument('-i', '--input', help='The dir path of input dataset', type=str, required=True)
    return parser.parse_args()

def process_file(filepath):
    with open(filepath, "r") as file:
        code = file.read()

    code = re.sub('(?<!:)\\/\\/.*|\\/\\*(\\s|.)*?\\*\\/', "", code).strip()

    with open(filepath, "w") as file:
        file.write(code)

    with open(filepath, "r") as file:
        nor_code = clean_gadget(file.readlines())

    with open(filepath, "w") as file:
        file.writelines(nor_code)

def normalize(path):
    for setfolder in os.listdir(path):
        for catefolder in os.listdir(os.path.join(path, setfolder)):
            filepath = os.path.join(path, setfolder, catefolder)
            print(catefolder)
            process_file(filepath)

def main():
    args = parse_options()
    normalize(args.input)

if __name__ == '__main__':
    main()
