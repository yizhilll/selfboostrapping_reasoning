import json
import orjsonl
import requests
import time
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
import ray
import numpy as np
from copy import deepcopy
# from openrlhf.utils.logging_utils import init_logger
import pickle
from glob import glob
import os
import gc
from datasets import load_from_disk, Dataset
from typing import TypeVar, Dict, List, Optional, Union, Any, Tuple, Literal
import multiprocessing
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from collections import defaultdict
from pydantic_settings import BaseSettings
import re
import signal

# import sys
# sys.path.append('/aifs4su/mmcode/codeclm/o1')

# from evaluation.parser import extract_answer, find_box
# from evaluation.grader import math_equal_process, math_equal
from evaluation.verifier_math_conversion import extract_label_content, remove_outer_parentheses, is_valid_math
from evaluation.code import extract_code_and_test
from math_verify import parse, verify, LatexExtractionConfig
from latex2sympy2_extended import NormalizationConfig
# from openrlhf.utils.data_record_util import JsonlFileHandler

from multiprocessing import Process, Queue

# logger = init_logger(__name__)

MAX_INS_LEN_AS_KEY = 2400

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Process timeout")

def find_box(pred_str: str):
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a

class Verifier_Settings(BaseSettings):
    """Verifier_Settings class that handles configuration with environment variable override support."""
    pt_model_name: str = "OpenO1/OpenO1-Qwen-7B-v0.1"
    datapath_list: str = "/map-vepfs/openo1/LLaMA-Factory/data/numina.jsonl"
    qa_mapping_file: str = "evaluation/data/templated_numina_qa_mapping.pkl"
    num_verifier_processes: int = 1
    prompt_max_len: int = 1024
    input_key: str = "query"
    output_key: str = "response"
    answer_key: str = 'None'
    maximum_len_reward: float = -1.0
    reward_ref_format: str = 'output_only'
    apply_chat_template: bool = True
    return_additional_info: bool = True
    save_query_response: bool = True
    save_reward_records_folder: str = 'None'
    save_record_batch_size: int = 256
    verifier_base_url: str = 'http://127.0.0.1:15078/get_response_from_llm'
    local_verifier_path: str = 'None'
    tp: int = 8
    voting_k: int = 1
    use_vllm_group: bool = False
    enable_math_expr_extract: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._override_from_env()
        
    @classmethod
    def _get_env_variables(cls) -> Dict[str, Any]:
        env_settings = {}
        settings_fields = cls.__annotations__
        
        for env_var in os.environ:
            if env_var.isupper():
                setting_name = env_var.lower().removeprefix('rm_')
                if setting_name in settings_fields:
                    expected_type = settings_fields[setting_name]
                    value = os.environ[env_var]
                    try:
                        if expected_type == bool:
                            value = value.lower() in ('true', '1', 'yes')
                        else:
                            value = expected_type(value)
                        env_settings[setting_name] = value
                    except ValueError as e:
                        print(f"Warning: Could not convert environment variable {env_var} to type {expected_type}: {e}")
        return env_settings

    def _override_from_env(self) -> None:
        env_settings = self._get_env_variables()
        if env_settings:
            print("Overriding settings with environment variables:")
            for key, value in env_settings.items():
                if hasattr(self, key):
                    print(f"  {key}: {getattr(self, key)} -> {value}")
                    setattr(self, key, value)

# Utility functions
def is_number(element: str) -> bool:
    try:
        float(element.replace(" ", ""))
        return True
    except ValueError:
        return False

def split_output(text: str) -> Tuple[str, str]:
    cot = ""
    output = text.strip()
    
    # Try to find the final answer after "Therefore" or similar keywords
    keywords = ['<Output>']
    for keyword in keywords:
        if keyword.lower() in output.lower():
            parts = output.lower().split(keyword.lower())
            if len(parts) > 1:
                cot = output[:output.lower().rindex(keyword.lower())]
                output = output[output.lower().rindex(keyword.lower()):].strip()
                break
    
    return cot, output

def extract_instruction_qwen(text: str) -> str:
    try:
        prefix = '<|im_start|>user\n'
        suffix = '<|im_end|>\n<|im_start|>assistant'
        start_pos = text.index(prefix) + len(prefix)
        try:
            end_pos = text.index(suffix, start_pos)
        except ValueError:
            end_pos = len(text)
        text = text[start_pos:end_pos]
    except:
        print(f'Could not extract instruction from:', text)
    return text

def extract_instruction_llama(text: str) -> str:
    text = text.split('<|eot_id|><|start_header_id|>assistant<|end_header_id|>')[0].strip()
    text = text.split('<|start_header_id|>user<|end_header_id|>')[1].strip()
    return text

def extract_response_qwen(text: str) -> str:
    prefix = '<|im_start|>assistant'
    suffix = '<|im_end|>'
    try:
        start_pos = text.index(prefix) + len(prefix)
    except:
        try:
            start_pos = text.index('<Thought>')
        except:
            start_pos = 0 
    try:
        end_pos = text.index(suffix, start_pos)
    except ValueError:
        end_pos = len(text)
    return text[start_pos:end_pos]

def extract_response_llama(text: str) -> str:
    return text.split('<|start_header_id|>assistant<|end_header_id|>')[1].strip()

def format_instruction(text: str) -> str:
    return text[:MAX_INS_LEN_AS_KEY]


UNSUPPORTED_DATA_TYPES = [
    'no_type',
    # ('Math', 'b'),
    ('Math', 'c'),
    ('Reasoning', 'b'),
    # ('mcq_26m_close_form', 'b'),
]

def get_instruction_type_mapping(entries: List[Dict]) -> Tuple[Dict[Tuple[str, str], List[int]], Dict[int, Tuple[str, str]]]:
    type_to_ids = defaultdict(list)
    ids_to_type = {}
    
    for idx, entry in enumerate(entries):
        # print(idx, entry)
        # entry = qa_mapping.get(ins)
        
        if entry is not None:
            if entry['source'] == 'mcq_26m' and entry['answer_type'] in ['a', 'b', 'c', 'd', 'e']:
                key = ('mcq_26m', entry['answer_type'])
            elif entry['source'] == 'mcq_26m_close_form' and entry['answer_type'] in ['a', 'b', 'c']:
                key = ('mcq_26m_close_form', entry['answer_type'])
            else:
                key = (entry['Topic'], entry['answer_type'])
        else:
            key = 'no_type'
            
        type_to_ids[key].append(idx)
        ids_to_type[idx] = key
    
    for key in UNSUPPORTED_DATA_TYPES:
        assert key not in type_to_ids, f"Unsupported data type: {key}"
    
    return type_to_ids, ids_to_type



from tenacity import retry, stop_after_attempt, wait_fixed
from tenacity import retry_if_exception_type, stop_after_delay

# 修改 rule_based_verifier 函数
# @retry(
#     stop=(
#         # stop_after_attempt(3) &  # 最多重试3次
#         stop_after_delay(3)      # 总执行时间不超过3秒
#     ),
#     # wait=wait_fixed(0.01),          # 重试间隔1秒
#     retry=retry_if_exception_type(Exception),
#     # reraise=True
# )
# def wrap_latex_text(s):
#     if 'text{' in s:
#         try:
#             prefix = 'text{'
#             suffix = '<|im_end|>\n<|im_start|>assistant'
#             start_pos = text.index(prefix) + len(prefix)
#             try:
#                 end_pos = text.index(suffix, start_pos)
#             except ValueError:
#                 end_pos = len(text)
#             text = text[start_pos:end_pos]
#         except:
#             print(f'Could not extract instruction from:', text)
#         return text
#     else:
#         return s

@ray.remote(max_retries=1)
def rule_based_verifier_ray(
    output: str, 
    gold_standard: str,
    enable_math_expr_extract: bool
) -> float:
    return rule_based_verifier(output=output, gold_standard=gold_standard, enable_math_expr_extract=enable_math_expr_extract)

def rule_based_verifier(
    output: str, 
    gold_standard: str,
    enable_math_expr_extract: bool
) -> float:
    cot, output = split_output(output) # not extrac the boxed at this step
    
    # all in math_verify!
    
    # this could first check on the string match
    answer_string = find_box(output) # find the actual response and remove the boxed{}
    if gold_standard == answer_string:
        return 1.0
    if gold_standard.startswith('$') and gold_standard.endswith('$'):
        if gold_standard[1:-1] == answer_string:
            return 1.0
    # if answer_string.startswith('{') and answer_string.endswith('}') \
    #     and answer_string[1:-1] == gold_standard:
    #         return 1.0
    
    answer_parsed = parse(
                output , # the parse will need to extract the boxed
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            # equations=True,
                            boxed=True,
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
                parsing_timeout=3,
            )
    answer_string_parsed = parse(
                "\\boxed{" + answer_string + "}",  # the parse will need to extract the boxed
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            # equations=True,
                            boxed=True,
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    ),
                ],
                extraction_mode="first_match",
                parsing_timeout=3,
            )
    # preprend parsing sourounding for gold_standards
    if not(gold_standard.startswith('$') and gold_standard.endswith('$')) and not(gold_standard.startswith('\\boxed{') and gold_standard.endswith('}')):
        gold_standard = "\\boxed{" + gold_standard + "}"
    gold_parsed = parse(gold_standard, extraction_mode="first_match", parsing_timeout=3)
    if len(gold_parsed) != 0:
        # print('## first match', float(verify(answer_parsed, gold_parsed)))
        # print('gold_parsed:', gold_parsed)
        # print('answer_parsed:', answer_parsed, ". from:", output[-128:])
        return float(verify(gold_parsed, answer_parsed) or verify(gold_parsed, answer_string_parsed))
    # else:
    #     gold_parsed = parse(f'${gold_standard}$', extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
    #     # print('## second match', float(verify(answer_parsed, gold_parsed)))
    #     # print('gold_parsed:', gold_parsed)
    #     # print('answer_parsed:', answer_parsed)
    #     if len(gold_parsed) != 0:
    #         return float(verify(gold_parsed, answer_parsed) or verify(gold_parsed, answer_string_parsed))
    #     else:
    #         assert ValueError, f"Invalid gold standard: {gold_standard}, which should be filtered before training"
    
    
    # if is_number(output):
    #     # return float(math_equal_process((output, gold_standard)))
    #     return float(verify(parse(gold_standard), parse(output)))

    # prediction = extract_answer(pred_str=output, data_name='N/A', use_last_number=False)

    # if prediction:
    #     # Qwen-Sympy implementation
    #     return float(math_equal(prediction=prediction, reference=gold_standard))
        # HF math_verify implementation
        # return float(verify(parse(gold_standard), parse(prediction)))

    # if enable_math_expr_extract:
    #     is_label, label, content = extract_label_content(output)
    #     if is_label:
    #         output = content    
    #     output = remove_outer_parentheses(output)
        
    #     is_math_expr, type_, output_result = is_valid_math(output)
    #     is_ref_math_expr, type_, ref_result = is_valid_math(gold_standard)
    #     if is_math_expr and is_ref_math_expr:
    #         is_equal = math_equal_process((output_result, ref_result))
    #         return float(is_equal)

    return 0.0



# 修改 safe_rule_based_verifier 函数
def safe_rule_based_verifier(
    output: str, 
    gold_standard: str, 
    enable_math_expr_extract: bool,
    retry: int = 1,
    timeout_per_try: int = 3,
) -> float:
    current_try = 0
    while current_try < retry:
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_per_try) # `timeout_per_try` must be integer
            reward = rule_based_verifier(output=output, gold_standard=gold_standard, enable_math_expr_extract=enable_math_expr_extract)
            signal.alarm(0)
            return reward
        except Exception as e:
            print(f'Trying rule_based_verifier() {current_try+1}: running over {timeout_per_try} seconds, killed by timeout error')
            print(f'error output:', e)
            current_try += 1
    
    print(f"[WARNING]: Rule-based verifier failed after {retry} attempts with timeout")
    return 0.0




def get_math_rewards(
    ref_responses: List[str],
    pred_responses: List[str],
    enable_math_expr_extract: bool,
    use_ray: bool = True
) -> List[float]:

    if use_ray:
        refs = []
        for pred, ref in zip(pred_responses, ref_responses):
            refs.append(rule_based_verifier_ray.remote(output=pred, gold_standard=ref, enable_math_expr_extract=enable_math_expr_extract))
        rewards = []
        for i, ref in enumerate(refs):
            try:
                rewards.append(ray.get(ref, timeout=1200))
                # rewards.append(ray.get(ref))
            except (ray.exceptions.RayActorError,
                   ray.exceptions.GetTimeoutError,
                   ray.exceptions.RaySystemError,
                   ray.exceptions.ObjectLostError,
                   ray.exceptions.TaskCancelledError,
                   ray.exceptions.RayTaskError,
                   Exception) as e:
                print(f"can't verify {pred_responses[i]}, set reward 0, failed with error: {str(e)}")
                rewards.append(0.0)

    else:
        rewards = []
        for pred, ref in zip(pred_responses, ref_responses):
            reward = safe_rule_based_verifier(output=pred, gold_standard=ref, enable_math_expr_extract=enable_math_expr_extract)
            
            rewards.append(reward)
    return rewards

def safe_code_verifier(idx, entry, extracted_pred, retry=1, timeout_per_try=3) -> float:
    current_try = 0
    while current_try < retry:
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_per_try) # `timeout_per_try` must be integer
            idx, reward = extract_code_and_test((idx, entry, extracted_pred))
            signal.alarm(0)
            return idx, reward
        except:
            print(f'Trying extract_code_and_test() {current_try+1}: running over {timeout_per_try} seconds, killed by timeout error')
            current_try += 1
    
    print(f"[WARNING]: Rule-based extract_code_and_test verifier failed after {retry} attempts with timeout")
    return idx, 0.0

def code_verifier(idx, entry, extracted_pred) -> float:
    idx, reward = extract_code_and_test((idx, entry, extracted_pred))
    return idx, reward

@ray.remote(max_retries=1)
def code_verifier_ray(idx, entry, extracted_pred) -> float:
    return code_verifier(idx, entry, extracted_pred)


def get_code_rewards(entries: List[Dict], pred_responses: List[str], use_ray: bool = True) -> List[float]:
    rewards = []
    if use_ray:
        refs = []
        for idx, (entry, pred) in enumerate(zip(entries, pred_responses)):
            cot, extracted_pred = split_output(pred)
            extracted_pred = find_box(extracted_pred)
            refs.append(code_verifier_ray.remote(idx, entry, extracted_pred))
        rewards = []
        for i, ref in enumerate(refs):
            try:
                idx, reward = ray.get(ref, timeout=1200)
                # idx, reward = ray.get(ref)
                rewards.append(reward)
            except (ray.exceptions.RayActorError,
                   ray.exceptions.GetTimeoutError,
                   ray.exceptions.RaySystemError,
                   ray.exceptions.ObjectLostError,
                   ray.exceptions.TaskCancelledError,
                   ray.exceptions.RayTaskError,
                   Exception) as e:
                print(f"can't verify: ...{pred_responses[i][-128:]}. Set reward 0, failed with error: {str(e)}")
                rewards.append(0.0)
    else:
        for idx, (entry, pred) in enumerate(zip(entries, pred_responses)):
            cot, extracted_pred = split_output(pred)
            extracted_pred = find_box(extracted_pred)
            idx, reward = safe_code_verifier(idx, entry, extracted_pred)
            rewards.append(reward)
    return rewards

# Core reward calculation functions
def process_mcq_reward(answer_type: str, pred: str, option: str, content: str) -> float:

    patterns = {
        'a': r'(.*?)', # dummy placeholder
        'b': r'<option>\[(.*?)\]</option>',
        'c': r'<<\((.*?)\)>>',
        'd': r'==(.*?)==',
        'e': r'##(.*?)##'
    }
    
    possible_gt = [option, option+'.']
    # if answer_type != 'c':
    possible_gt.extend([content, f'{option}. {content}', f'{option}.{content}'])
    
    if answer_type == 'a':
        pred_content = pred.strip().rstrip()
    else:
        pattern = patterns.get(answer_type)
        if not pattern:
            return 0.0
        match = re.search(pattern, pred, re.IGNORECASE)
        pred_content = match.group(1) if match else ""
        pred_content = pred_content.strip().rstrip()
    return float(pred_content in possible_gt)
    
def get_matching_rewards(entries: List[Dict], pred_responses: List[str]) -> List[float]:
    rewards = []
    
    for idx, (entry, pred) in enumerate(zip(entries, pred_responses)):
        ground_truth = entry['response']
        cot, output = split_output(pred)
        # output = extract_answer(pred_str=output, data_name='N/A', use_last_number=False)
        output = find_box(output)
        
        ground_truth = ground_truth.strip().rstrip()
        output = output.strip().rstrip()
        is_label, label, labelled_content = extract_label_content(output)
        if is_label:
            labelled_content.strip().rstrip()
        if (labelled_content == ground_truth) or (output == ground_truth):
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards

def get_mcq_rewards(entries: List[Dict], pred_responses: List[str]) -> List[float]:
    rewards = []
    
    for idx, (entry, pred) in enumerate(zip(entries, pred_responses)):
        ground_truth = entry['response']
        cot, output = split_output(pred)
        # output = extract_answer(pred_str=output, data_name='N/A', use_last_number=False)
        output = find_box(output)

        option = None
        ground_truth_content = None
        for i in range(5):
            if ground_truth.startswith(chr(ord('A')+i)):
                option = chr(ord('A')+i)
                ground_truth_content = ground_truth.removeprefix(option+'.').strip().rstrip()
                break
        reward = process_mcq_reward(
            entry['answer_type'], 
            output, 
            option, 
            ground_truth_content
        )
        rewards.append(reward)

    return rewards

def process_queries(
    queries: List[str], 
    model_type: str, 
    output_key: str,
    enable_math_expr_extract: bool,
    qa_mapping_subset: Dict = None,
    entries: List[Dict] = None,
    use_ray: bool = True
) -> Dict:
    # Extract instructions based on model type
    if model_type == 'llama':
        instructions = [format_instruction(extract_instruction_llama(q)) for q in queries]
        pred_responses = [extract_response_llama(q) for q in queries]
    elif model_type == 'qwen':
        instructions = [format_instruction(extract_instruction_qwen(q)) for q in queries]
        pred_responses = [extract_response_qwen(q) for q in queries]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    assert qa_mapping_subset is not None or entries is not None
    if entries is None:
        entries = [deepcopy(qa_mapping_subset[ins]) for ins in instructions] # ensure the instruction is in the qa_mapping_subset
    
    type_to_ids, _ = get_instruction_type_mapping(entries=entries)
    assert 'no_type' not in type_to_ids
    
    for i in range(len(entries)):
        entries[i]['model_response'] = pred_responses[i]
    
    return process_entries(
        entries=entries,
        output_key=output_key,
        model_output_key='model_response',
        enable_math_expr_extract=enable_math_expr_extract,
        use_ray=use_ray
    )

def process_entries(
    entries: List[Dict],
    enable_math_expr_extract: bool,
    use_ray: bool = True,
    output_key: str = 'response',
    model_output_key: str = 'model_response',
) -> Dict:

    type_to_ids, _ = get_instruction_type_mapping(entries=entries)
    assert 'no_type'not in type_to_ids
    
    # pred_responses = [entry[model_output_key] for entry in entries]
    rewards = np.zeros(len(entries))
    
    # Handle math equations
    math_indices = type_to_ids[('Math', 'a')] + \
        type_to_ids[('Math', 'b')] + \
                   type_to_ids[('Reasoning', 'a')] + \
                   type_to_ids[('mcq_26m_close_form', 'a')] + \
                   type_to_ids[('mcq_26m_close_form', 'b')]
                
    if math_indices:
        math_rewards = get_math_rewards(
            ref_responses=[entries[i][output_key] for i in math_indices],
            pred_responses=[entries[i][model_output_key] for i in math_indices],
            enable_math_expr_extract=enable_math_expr_extract,
            use_ray=use_ray,
        )
        rewards[math_indices] = math_rewards

    # Handle code evaluation
    code_indices = type_to_ids[('Code', 'assert')] + type_to_ids[('Code', 'input')]
    if code_indices:
        code_rewards = get_code_rewards(
            entries=[entries[i] for i in code_indices],
            pred_responses=[entries[i][model_output_key] for i in code_indices],
            use_ray=use_ray,
        )
        rewards[code_indices] = code_rewards

    # Handling MCQ rewards
    mcq_indices = []
    for i in range(5):
        mcq_indices.extend(type_to_ids[('mcq_26m', chr(ord('a')+i))])
    if mcq_indices:
        mcq_rewards = get_mcq_rewards(
            entries=[entries[i] for i in mcq_indices],
            pred_responses=[entries[i][model_output_key] for i in mcq_indices],
        )
        rewards[mcq_indices] = mcq_rewards

    # Handle matching cases
    matching_indices = []
    for entry_id in type_to_ids[('Reasoning', 'a')]:
        if rewards[entry_id] == 0.0: # not matched by math-verify
            matching_indices.append(entry_id)
    # matching_indices.extend(type_to_ids[('Reasoning', 'a')])
    if matching_indices:
        matching_rewards = get_matching_rewards(
            entries=[entries[i] for i in matching_indices],
            pred_responses=[entries[i][model_output_key] for i in matching_indices],
        )
        rewards[matching_indices] = matching_rewards
    
    # TODO: support LLM-as-judge in the future
    
    

    return {
        "rewards": rewards.tolist(),
        "correctness_rewards": rewards.tolist()
    }
    
# 首先添加一个函数来确保字典内容可序列化
def ensure_serializable(obj):
    """确保对象是可序列化的基本类型"""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {k: ensure_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [ensure_serializable(x) for x in obj]
    else:
        # 对于其他类型，尝试转换为字符串或基本类型
        try:
            return str(obj)
        except:
            return None

def is_legal_cot_split_boxed(s):
    # Required substrings in order
    # First, let's check the main tags are in order and appear once
    required_tags = ['<Thought>', '</Thought>', '<Output>', '</Output>']
    last_pos = -1
    
    # Check order and uniqueness of main tags
    for tag in required_tags:
        pos = s.find(tag)
        
        # Check if tag exists
        if pos == -1:
            return False, f"Missing tag: {tag}"
        
        # Check if it appears after the previous tag
        if pos <= last_pos:
            return False, f"Incorrect order: {tag} should appear after previous tag"
        
        # Check if tag appears more than once
        if s.count(tag) > 1:
            return False, f"Tag appears multiple times: {tag}"
        
        last_pos = pos

    # Now, let's check the 'boxed' requirement within Output tags
    # Extract content between Output tags
    output_start = s.find('<Output>') + len('<Output>')
    output_end = s.find('</Output>')
    output_content = s[output_start:output_end]
    boxed_count = output_content.count('boxed')
    if boxed_count == 0:
        return False, "'boxed' is missing within Output tags"
    elif boxed_count > 1:
        return False, "'boxed' appears multiple times within Output tags"
    
    return True, "String is legal"

class ResponseVerifier:
    def __init__(self, settings: Verifier_Settings):
        self.settings = settings
        self.process_pool = ProcessPoolExecutor(max_workers=settings.num_verifier_processes)
        self.qa_mapping = None
        self.tokenizer = None
        self.model_type = None
        self.local_llm = None
        self.initialize()

    def initialize(self, build_qa_mapping: bool = True):
        print('Initializing verifier with settings:\n', self.settings)
        self.tokenizer = AutoTokenizer.from_pretrained(self.settings.pt_model_name, trust_remote_code=True)

        model_name_lower = self.settings.pt_model_name.lower()
        if 'llama' in model_name_lower:
            self.model_type = 'llama'
        elif 'qwen' in model_name_lower:
            self.model_type = 'qwen'
        else:
            raise ValueError(f"Unsupported model type in pt_model_name: {self.settings.pt_model_name}")

        if build_qa_mapping:
            self._init_qa_mapping()
        self._init_save_directory()
        self._init_local_llm()
        
        self.record_batch = []

    def _init_qa_mapping(self):
        if not self._is_none_like(self.settings.qa_mapping_file) and os.path.exists(self.settings.qa_mapping_file):
            print(f'Loading qa_mapping from {self.settings.qa_mapping_file}')
            with open(self.settings.qa_mapping_file, 'rb') as f:
                self.qa_mapping = pickle.load(f)
        else:
            self.qa_mapping = self._build_answer_mapping()
            if not self._is_none_like(self.settings.qa_mapping_file):
                os.makedirs(os.path.dirname(self.settings.qa_mapping_file), exist_ok=True)
                with open(self.settings.qa_mapping_file, 'wb') as f:
                    pickle.dump(self.qa_mapping, f)
        ins, item = next(iter(self.qa_mapping.items()))
        print(f'example qa_mapping:\ninstruction:{ins}\nitem:{item}')
        
    def _init_save_directory(self):
        if not self._is_none_like(self.settings.save_reward_records_folder):
            print(f'Saving reward records to {self.settings.save_reward_records_folder}')
            os.makedirs(self.settings.save_reward_records_folder, exist_ok=True)

    def _init_local_llm(self):
        if not self._is_none_like(self.settings.local_verifier_path):
            if self.settings.use_vllm_group:
                self.local_llm = LLMs(
                    self.settings.local_verifier_path,
                    tensor_parallel_size=self.settings.tp,
                    enable_prefix_caching=True,
                    trust_remote_code=True,
                )
            else:
                self.local_llm = LLM(
                    model=self.settings.local_verifier_path,
                    tensor_parallel_size=self.settings.tp,
                    enable_prefix_caching=True,
                    trust_remote_code=True,
                )

    def _is_none_like(self, value: str) -> bool:
        return value is None or value.lower() == "none" or value == ""

    def _encode_decode_item(self, item: Dict, model_response_key: str) -> Dict:
        encoded_prompt = self.tokenizer(
            item['query'],
            return_tensors="pt",
            add_special_tokens=False,
            max_length=self.settings.prompt_max_len,
            padding=True,
            truncation=True,
        )
        decoded_prompt = self.tokenizer.decode(encoded_prompt['input_ids'][0], skip_special_tokens=False)

        if self.settings.apply_chat_template:
            messages = [
                {"role": "user", "content": decoded_prompt},
                {"role": "assistant", "content": item[model_response_key]},
            ]
            item['decoded_prompt'] = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            item['decoded_prompt'] = decoded_prompt
        return item

    def _build_answer_mapping(self) -> Dict[str, Any]:
        """
        Builds a mapping between formatted instructions and their corresponding answers.
        This function processes input data to create a standardized mapping that can be 
        used for response verification.
        
        Returns:
            Dict[str, Any]: A dictionary mapping formatted instructions to their full data entries
        """
        # Helper function to tokenize text with configurable parameters
        def tokenize_fn(texts: Union[str, List[str]], max_length: int = 1024, 
                    padding: bool = True) -> Dict[str, torch.Tensor]:
            """
            Tokenizes input texts using the class tokenizer with specified parameters.
            Handles both single texts and batches, with optional padding.
            """
            if not padding:
                # For non-padded tokenization, return raw tokenized output
                return self.tokenizer(
                    texts,
                    add_special_tokens=False,
                    max_length=max_length,
                    truncation=True
                )
            
            # For padded tokenization, return tensor-based batch
            batch = self.tokenizer(
                texts,
                return_tensors="pt",
                add_special_tokens=False,
                max_length=max_length,
                padding=True,
                truncation=True
            )
            return {k: v for k, v in batch.items()}

        def preprocess_data(data: Dict[str, Any]) -> str:
            """
            Preprocesses input data based on configuration settings.
            Handles both string inputs and chat-format inputs.
            """
            if self.settings.apply_chat_template:
                # Handle chat format inputs
                chat = data[self.settings.input_key]
                if isinstance(chat, str):
                    # Convert single string to chat format
                    chat = [{"role": "user", "content": chat}]
                prompt = self.tokenizer.apply_chat_template(
                    chat, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                # Handle regular string inputs
                prompt = data[self.settings.input_key]
            return prompt

        def aliganed_openrlhf_process_queries(data: List[Dict[str, Any]]) -> List[str]:
            """
            Processes a list of queries through tokenization and formatting.
            Includes progress tracking and error handling.
            """
            # Process all prompts with progress tracking
            all_prompts = []
            for d in tqdm(data, desc='Processing prompts'):
                try:
                    processed = preprocess_data(d)
                    all_prompts.append(processed)
                except Exception as e:
                    logger.warning(f"Error processing prompt: {e}")
                    all_prompts.append("")  # Add empty string for failed processing
                    
            # Create dataset and encode queries
            ds = Dataset.from_list([{'query': p} for p in all_prompts])
            
            # Perform encoding and decoding with progress tracking
            encoded_queries = []
            for item in tqdm(ds, desc='Encoding instructions'):
                try:
                    encoded = tokenize_fn(
                        item['query'], 
                        max_length=self.settings.prompt_max_len
                    )
                    encoded_queries.append(encoded)
                except Exception as e:
                    logger.warning(f"Error encoding query: {e}")
                    encoded_queries.append(None)

            # Decode and format instructions
            decoded_queries = []
            for item in tqdm(encoded_queries, desc='Decoding instructions'):
                if item is not None:
                    try:
                        decoded = self.tokenizer.decode(
                            item['input_ids'][0], 
                            skip_special_tokens=False
                        )
                        
                        # formatted = self._format_instruction(
                        #     self._extract_instruction(decoded)
                        # )
                        if self.model_type == 'llama':
                            formatted = format_instruction(extract_instruction_llama(decoded))
                        else:
                            formatted = format_instruction(extract_instruction_qwen(decoded))
                        decoded_queries.append(formatted)
                    except Exception as e:
                        logger.warning(f"Error decoding query: {e}")
                        decoded_queries.append("")
                else:
                    decoded_queries.append("")

            return decoded_queries

        # Load and process all data
        data = []
        if isinstance(self.settings.datapath_list, str):
            datapaths = self.settings.datapath_list.split(',')
        else:
            datapaths = self.settings.datapath_list
            
        for filepath in datapaths:
            try:
                if filepath.endswith('.jsonl'):
                    data.extend(orjsonl.load(filepath))
                else:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data.extend(json.load(f))
            except Exception as e:
                logger.error(f"Error loading file {filepath}: {e}")
                continue

        # Process queries and build mapping
        formatted_queries = aliganed_openrlhf_process_queries(data)
        assert len(formatted_queries) == len(data), "Query processing altered data length"

        # Build instruction to answer mapping with duplicate tracking
        instruct_to_golden = {}
        stats = {
            'total_count': 0,
            'identical_count': 0,
            'duplicate_count': 0
        }

        for data_idx, item in tqdm(enumerate(data), desc='Building Q->A mapping'):
            formatted_q = formatted_queries[data_idx]
            stats['total_count'] += 1

            if formatted_q in instruct_to_golden:
                if instruct_to_golden[formatted_q].get('uid') == item.get('uid'):
                    stats['identical_count'] += 1
                else:
                    stats['duplicate_count'] += 1
                continue

            instruct_to_golden[formatted_q] = item

        # Log statistics and sample mapping
        logger.info(f"Mapping statistics:")
        logger.info(f"Total Q-A pairs: {stats['total_count']}")
        logger.info(f"Duplicate questions (different UIDs): {stats['duplicate_count']}")
        logger.info(f"Duplicate questions (same UID): {stats['identical_count']}")
        logger.info(f"Final mapping size: {len(instruct_to_golden)}")

        
        sample_key = next(iter(instruct_to_golden))
        logger.info("Sample QA mapping:")
        logger.info(f"Q: {sample_key}")
        logger.info(f"A: {instruct_to_golden[sample_key].get(self.settings.output_key, 'N/A')}")

        return instruct_to_golden

    def _load_and_filter_data(self, input_file: str) -> List[Dict]:
        if "*" not in input_file:
            if input_file.endswith('.jsonl'):            
                all_data = orjsonl.load(input_file)
            else:
                with open(input_file, 'r') as f:
                    all_data = json.load(f)
        else:
            all_data = []
            fp_list = glob(input_file)
            fp_list.sort()
            for fp in fp_list:
                if input_file.endswith('.jsonl'):   
                    all_data.extend(orjsonl.load(fp))
                else:
                    with open(fp, 'r') as f:
                        all_data.extend(json.load(f))

        return [x for x in all_data if self._is_keep_item(x)]

    def _is_keep_item(self, entry: Dict) -> bool:
        if entry['response'] == '' or entry['response'] is None:
            return False
            
        if entry['Topic'] == 'Math' and entry['answer_type'] in ['c']:
            return False

        if entry['Topic'] == 'Reasoning' and entry['answer_type'] in ['b']:
            return False    
            
        return True

    def _process_data_batches_save_to_local(
        self, 
        data_list: List[Dict], 
        output_path: str, 
        model_response_key: str, 
        save_reward_key: str,
        batch_size: int = 1000, 
        request_batch_size: int = 1
    ):
        ray.init()
        n_written = 0
        if os.path.exists(output_path + '.tmp'):
            with open(output_path + '.tmp', 'r') as f:
                for _ in f:
                    n_written += 1
            assert n_written % (batch_size * request_batch_size) == 0
            print(f'Skip {n_written} written data, continue on the rest')
            
        
        with tqdm(total=(len(data_list)-n_written)//(request_batch_size*batch_size)) as pbar:
            for i in range(n_written, len(data_list), batch_size):
                split_data = [self._encode_decode_item(item, model_response_key)['decoded_prompt'] 
                            for item in data_list[i:i+batch_size]]
                
                # Create qa_mapping subset with serializable content
                qa_mapping_subset = {}
                for prompt in split_data:
                    if self.model_type == 'llama':
                        instruction = format_instruction(extract_instruction_llama(prompt))
                    else:
                        instruction = format_instruction(extract_instruction_qwen(prompt))
                    # print('prompt:', prompt)
                    # print('instruction:', instruction)
                    try:
                        # Ensure the content is serializable
                        # qa_mapping_subset[instruction] = ensure_serializable(
                        #     {
                        #         'Topic': self.qa_mapping[instruction].get('Topic', ''),
                        #         'answer_type': self.qa_mapping[instruction].get('answer_type', ''),
                        #         'source': self.qa_mapping[instruction].get('source', ''),
                        #         self.settings.output_key: self.qa_mapping[instruction].get(self.settings.output_key, '')
                        #     }
                        # )
                        qa_mapping_subset[instruction] = self.qa_mapping[instruction]
                    except Exception as e:
                        print(f'error when getting response for instruction:', instruction[-256:])
                        print(f'error message:', str(e))
                        exit()
                
                try:
                    # Call ray remote function with minimal data
                    results = [process_queries_ray.remote(
                        queries=split_data,
                        model_type=self.model_type,
                        qa_mapping_subset=qa_mapping_subset,
                        output_key=self.settings.output_key,
                        enable_math_expr_extract=self.settings.enable_math_expr_extract
                    )]
                    results = ray.get(results)
                except Exception as e:
                    print(f"Error occurred: {str(e)}")
                    print("qa_mapping_subset keys:", list(qa_mapping_subset.keys())[:5])
                    print("Sample qa_mapping_subset value:", next(iter(qa_mapping_subset.values())) if qa_mapping_subset else None)
                    raise e
                
                with open(output_path + '.tmp', 'a', encoding='utf-8') as f:
                    for j, reward in enumerate(results[0]['correctness_rewards']):
                        data_list[i+j][save_reward_key] = reward
                        f.write(json.dumps(data_list[i+j], ensure_ascii=False) + '\n')
                        data_list[i+j] = None  # Clean up

                pbar.update(1)
                gc.collect()
        print(f'finish verifying, save to {output_path}')
        os.rename(output_path + '.tmp', output_path)

        ray.shutdown()


    def groundtruth_testing_local_results(
        self, 
        input_file: str, 
        task_id: int = 0, 
        total_split: int = 1,
        save_name: str = "", 
        output_dir: str = "",
        model_response_key: str = "response",
        save_reward_key: str = "verifier_score",
        use_qa_maaping: bool = True,       
    ):
        
        data_list = self._load_and_filter_data(input_file)
        
        # Split data if needed
        if total_split > 1:
            total_len = len(data_list)
            base_size = total_len // total_split
            remainder = total_len % total_split
            assert remainder == 0
            start_idx = task_id * base_size
            end_idx = start_idx + base_size
            data_list = data_list[start_idx:end_idx]
        
        # Setup output path
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            if save_name:
                output_path = os.path.join(output_dir, f'{save_name}_{task_id}_of_{total_split}.jsonl')
            else:
                output_path = os.path.join(output_dir, f'{task_id}_of_{total_split}.jsonl')
        
            if os.path.exists(output_path):
                n_exists = sum(1 for _ in open(output_path))
                if n_exists == len(data_list):
                    print(f"File {output_path} is complete, exiting")
                    return
        
        # Process data in batches
        if use_qa_maaping:
            self._process_data_batches_save_to_local(
                data_list=data_list,
                output_path=output_path,
                model_response_key=model_response_key,
                save_reward_key=save_reward_key
            )
        else:
            self._process_data_batches_save_to_local_no_mapping(
                data_list=data_list,
                output_path=output_path,
                model_response_key=model_response_key,
                save_reward_key=save_reward_key
            )

    def get_data_other_info(self, instructions: list[str]):
        info_dict = defaultdict(list)
        for ins in instructions:
            new_record = deepcopy(self.qa_mapping[ins])
            for key in ["difficulty", "quality", "Topic"]:
                info_dict[key].append(new_record[key])
            info_dict['uid'].append(new_record['uid'])
            info_dict['response'].append(new_record['response'])
        return info_dict

    def verify_query_on_the_fly(self, 
                                queries: List[str], 
                                pattern_mode: Literal['boxed', 'strict_boxed', 'cot_split_boxed'] = 'strict_boxed',
                                boxed_reward=False, 
                                use_ray=True,
                                print_verifier_example=False,
                                ) -> List[Dict]:

        # result = process_queries(queries, self.model_type, self.settings.output_key, self.settings.enable_math_expr_extract, qa_mapping_subset=self.qa_mapping, use_ray=use_ray)

        # Extract instructions based on model type
        if self.model_type == 'llama':
            instructions = [format_instruction(extract_instruction_llama(q)) for q in queries]
            pred_responses = [extract_response_llama(q) for q in queries]
        elif self.model_type == 'qwen':
            instructions = [format_instruction(extract_instruction_qwen(q)) for q in queries]
            pred_responses = [extract_response_qwen(q) for q in queries]
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        entries = []
        for ins, pred in zip(instructions, pred_responses):
            entry = deepcopy(self.qa_mapping[ins])
            entry['model_response'] = pred
            entries.append(entry)
            
        result = process_entries(
            entries=entries,
            output_key=self.settings.output_key,
            model_output_key='model_response',
            enable_math_expr_extract=self.qa_mapping,
            use_ray=use_ray
        )
        
        if self.settings.return_additional_info:
            # other_info = self.get_data_other_info(instructions)
            # result.update(other_info)
            for key in ["difficulty", "quality", "Topic", 'uid', 'response']:
                result[key] = []
                for entry in entries:
                    result[key].append(entry[key])
        
        # reset the rewards
        if boxed_reward:
            result['is_boxed'] = []
            for i in range(len(result['rewards'])):
                
                # strict boxed reward
                if pattern_mode == 'strict_boxed':
                    if pred_responses[i].count('boxed')==1:
                        result['is_boxed'].append(1)
                        if result['correctness_rewards'][i] > 0.0:
                            result['rewards'][i] = 1.0
                        else:
                            result['rewards'][i] = -0.5
                    else:
                        result['is_boxed'].append(0)
                        result['rewards'][i] = -1.0
                # simpleRL reward
                elif pattern_mode == 'boxed':
                    if result['correctness_rewards'][i] > 0.0:
                        result['is_boxed'].append(1)
                    else:
                        if 'boxed' not in pred_responses[i]:
                            result['rewards'][i] = -1.0
                            result['is_boxed'].append(0)
                        else:
                            result['rewards'][i] = -0.5
                            result['is_boxed'].append(1)
                elif pattern_mode == 'cot_split_boxed':
                    is_legal, illegal_reason = is_legal_cot_split_boxed(pred_responses[i])
                    result ['is_boxed'].append(int(is_legal))
                    if result['correctness_rewards'][i] > 0.0 and is_legal:
                        result['rewards'][i] = 1.0
                    # elif result['correctness_rewards'][i] > 0.0 and not is_legal:
                    #     result['rewards'][i] = 0.0
                    elif result['correctness_rewards'][i] <= 0.0 and is_legal:
                        result['rewards'][i] = -0.5
                    else:
                        result['rewards'][i] = -1.0
                else:
                    raise NotImplementedError(f"Unsupported pattern_mode: {pattern_mode}")

        if not self._is_none_like(self.settings.save_reward_records_folder):
            # build a list of dictionary for saving
            # result_new = deepcopy(result)
            result_new = [
                {key: values[i] for key, values in result.items()}
                for i in range(len(result['rewards']))
            ]
            if self.settings.save_query_response:
                for i in range(len(result_new)):
                    result_new[i]['query'] = instructions[i]
                    result_new[i]['model_response'] = pred_responses[i]
            if print_verifier_example:
                print(f'example verifier result:\n{result_new[0]}')
            # save_query_response
            # print(f'saving {len(result_new)} reward records to program cache')
            self.record_batch.extend(result_new)
            
            if len(self.record_batch) >= self.settings.save_record_batch_size:
                # this class generates a file name with uuid and time stamp
                # print(f'saving {len(self.record_batch)} reward records to {self.settings.save_reward_records_folder} and clean cache')
                # print('saving example:\n', self.record_batch[0])
                j_handle = JsonlFileHandler(base_dir=self.settings.save_reward_records_folder)
                j_handle.save_compressed_jsonl(self.record_batch)
                self.record_batch = []
        # print('===== example verifier output =====')
        # for k in result:
        #     print(f'{k}: {result[k][0]}')
        # print('instruction:', instructions[0])
        # print('model_response:', pred_responses[0])
        return result

def main(
    input_file, 
    output_dir,
    task_id=0,
    total_split=1,
    save_name="",
    model_response_key="short_cot",
    save_reward_key="verifier_score"
    ):
    settings = Verifier_Settings()
    verifier = ResponseVerifier(settings)
    verifier.groundtruth_testing_local_results(
        input_file=input_file,
        task_id=task_id,
        total_split=total_split,
        save_name=save_name,
        output_dir=output_dir,
        model_response_key=model_response_key,
        save_reward_key=save_reward_key
    )

from fire import Fire
if __name__ == '__main__':
    Fire(main)
