import unittest
import io
import sys
import json
import ast
import math
from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json
from tqdm import tqdm
sys.setrecursionlimit(6*10**5)

import time
def extract_code(s, language='python'):
    # Remove the ```json and ``` markers
    code_str = s.strip('`').removeprefix(f'{language}\n').removesuffix('\n')
    # Parse the JSON string
    return code_str

def safe_exec(code_str, globals=None, locals=None):
    # 捕获标准输出
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()  # 重定向到内存流

    try:
        exec(code_str, globals, locals)  # 执行代码
    except Exception as e:
        raise e  # 如果需要处理异常，可以捕获并重新抛出
    finally:
        # 恢复标准输出
        sys.stdout = original_stdout
def run_tests_inputs(code_str, test_cases, expected_outputs, id):
    class CodeTest(unittest.TestCase):
        pass

    def create_test(input_data, expected_output, id):
        def test(self):
            sys.stdin = io.StringIO(input_data)
            captured_output = io.StringIO()
            sys.stdout = captured_output

            code_globals = {'__name__': '__main__', '__builtins__': __builtins__}

            try:
                exec(code_str, code_globals)  # execute the code with the correct globals
            except Exception as e:
                self.fail(f"{id} 代码执行出错: {e}")

            sys.stdout = sys.__stdout__
            sys.stdin = sys.__stdin__

            output = captured_output.getvalue().strip()
            self.assertIn(output, expected_output)
        return test

    for i, (input_data, expected_output) in enumerate(zip(test_cases, expected_outputs)):
        test_method = create_test(input_data, expected_output, id)
        test_method.__name__ = f'test_case_{i+1}'
        setattr(CodeTest, test_method.__name__, test_method)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CodeTest)

    # Run tests
    # runner = unittest.TextTestRunner(stream=io.StringIO())
    runner = unittest.TextTestRunner(io.StringIO())
    result = runner.run(suite)
    return result.wasSuccessful()

def run_tests_assert(code_str, test_assertions):
    class CodeTest(unittest.TestCase):
        pass

    code_globals = {'__name__': '__main__', '__builtins__': __builtins__}
    try:
        exec(code_str, code_globals)
    except Exception as e:
        # print(f"代码定义出错: {e}")
        # print(f"错误代码\n{code_str}")
        return False

    def create_test_method(assertion_code, idx):
        def test_method(self):
            try:
                exec(assertion_code, code_globals)
            except AssertionError as e:
                self.fail(f"断言失败: {e}")
            except Exception as e:
                self.fail(f"执行断言时出错: {e}")
        return test_method

    for idx, assertion_code in enumerate(test_assertions):
        test_method = create_test_method(assertion_code, idx)
        test_method_name = f'test_case_{idx+1}'
        setattr(CodeTest, test_method_name, test_method)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CodeTest)

    # Run tests
    runner = unittest.TextTestRunner(io.StringIO())
    result = runner.run(suite)
    return result.wasSuccessful()

# 示例使用
def test_inputs(code_string,test_cases,expected_outputs,id=None):

    all_passed = run_tests_inputs(code_string, test_cases, expected_outputs,id)
    if not all_passed:
        print(id)
        print(code_string)
        print(test_cases)
        print(type(expected_outputs),expected_outputs)
    return all_passed

def test_assert(code_string,test_assertions):
    
    all_passed = run_tests_assert(code_string, test_assertions)
    return all_passed

# from wrapt_timeout_decorator import timeout
# @timeout(10, use_signals=False)
def extract_code_and_test(args):
    # code_str,
    # case_type,
    # cases,
    # id=None,
    
    # extract from composed params
    idx, entry, pred = args
    
    cases = entry['cases']
    
    pred = extract_code(pred)
    if entry['answer_type']=='assert':
        return idx, float(test_assert(pred, cases))
    elif entry['answer_type'] == 'input':
        return idx, float(test_inputs(
            pred,
            ast.literal_eval(cases['input_output'])['inputs'],
            ast.literal_eval(cases['input_output'])['outputs'],
            idx))
    else:
        raise NotImplementedError
