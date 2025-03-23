# import datasets
import pandas as pd
from glob import glob
import orjsonl
from collections import defaultdict
import pickle
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
from functools import partial
from copy import deepcopy
# from typing import List, Dict, Tuple
# from copy import deepcopy
import os
import time
# from local_verifier import process_entries
import json
import orjsonl
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import TypeVar, Dict, List, Optional, Union, Any, Tuple, Literal
from evaluation.verifier_math_conversion import extract_label_content
from evaluation.code import extract_code_and_test
from math_verify import parse, verify, LatexExtractionConfig
from latex2sympy2_extended import NormalizationConfig
import re
import signal
import ray
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Queue, Process
import queue

# def run_with_timeout(func, args=(), kwargs={}, timeout=30):
#     def wrapper(q, *args, **kwargs):
#         try:
#             q.put(func(*args, **kwargs))
#         except Exception as e:
#             q.put(e)
    
#     q = Queue()
#     p = Process(target=wrapper, args=(q,)+args, kwargs=kwargs)
#     p.daemon = True  # 设置为守护进程
#     p.start()
#     # 修改终止逻辑
#     p.join(timeout)
#     if p.is_alive():
#         p.kill()  # 先kill再terminate
#         p.terminate()
#         p.join()
#         raise TimeoutError()
    
#     return q.get()

def run_with_timeout(func, args=(), kwargs={}, timeout=30):
    def wrapper(q, *args, **kwargs):
        try:
            result = func(*args, **kwargs)
            q.put(result)
        except Exception as e:
            q.put(e)
            
    # q = mp.Queue()
    manager = mp.Manager()
    q = manager.Queue()  # 替代直接使用 mp.Queue
    
    p = mp.Process(target=wrapper, args=(q,)+args, kwargs=kwargs)
    p.daemon = True  # 设置为守护进程
    p.start()
    
    try:
        result = q.get(timeout=timeout)
    except mp.queues.Empty:
        result = TimeoutError()
    
    # 双重终止保障
    if p.is_alive():
        p.kill()
    p.terminate()
    p.join()
    
    if isinstance(result, Exception):
        raise result
    return result

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Process timeout")

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
            if entry.get('source', 'unknown') == 'mcq_26m' and entry['answer_type'] in ['a', 'b', 'c', 'd', 'e']:
                key = ('mcq_26m', entry['answer_type'])
            elif entry.get('source', 'unknown') == 'mcq_26m_close_form' and entry['answer_type'] in ['a', 'b', 'c']:
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
    
    # answer_parsed = parse(
    #             output , # the parse will need to extract the boxed
    #             extraction_config=[
    #                 LatexExtractionConfig(
    #                     normalization_config=NormalizationConfig(
    #                         nits=False,
    #                         malformed_operators=False,
    #                         basic_latex=True,
    #                         # equations=True,
    #                         boxed=True,
    #                         units=True,
    #                     ),
    #                     # Ensures that boxed is tried first
    #                     boxed_match_priority=0,
    #                     try_extract_without_anchor=False,
    #                 )
    #             ],
    #             extraction_mode="first_match",
    #             parsing_timeout=5,
    #         )
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
                parsing_timeout=5,
            )
    # preprend parsing sourounding for gold_standards
    if not(gold_standard.startswith('$') and gold_standard.endswith('$')) and not(gold_standard.startswith('\\boxed{') and gold_standard.endswith('}')):
        gold_standard = "\\boxed{" + gold_standard + "}"
    gold_parsed = parse(gold_standard, extraction_mode="first_match", parsing_timeout=5)
    if len(gold_parsed) != 0:
        # print('## first match', float(verify(answer_parsed, gold_parsed)))
        # print('gold_parsed:', gold_parsed)
        # print('answer_parsed:', answer_parsed, ". from:", output[-128:])
        # return float(verify(gold_parsed, answer_parsed) or verify(gold_parsed, answer_string_parsed))
                return float(verify(gold_parsed, answer_string_parsed))
    # else:
    #     gold_parsed = parse(f'${gold_standard}$', extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
    #     # print('## second match', float(verify(answer_parsed, gold_parsed)))
    #     # print('gold_parsed:', gold_parsed)
    #     # print('answer_parsed:', answer_parsed)
    #     if len(gold_parsed) != 0:
    #         return float(verify(gold_parsed, answer_parsed) or verify(gold_parsed, answer_string_parsed))
    #     else:
    #         assert ValueError, f"Invalid gold standard: {gold_standard}, which should be filtered before training"
    

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
            # reward = safe_rule_based_verifier(output=pred, gold_standard=ref, enable_math_expr_extract=enable_math_expr_extract)
            reward = rule_based_verifier(output=pred, gold_standard=ref, enable_math_expr_extract=enable_math_expr_extract)
            
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
            # idx, reward = safe_code_verifier(idx, entry, extracted_pred)
            idx, reward = code_verifier(idx, entry, extracted_pred)
            rewards.append(reward)
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
    
def create_histogram_ascii(data, bins=10, width=50):
    """
    Create an ASCII histogram from a list of numbers.
    
    Parameters:
    data: list of numbers
    bins: number of bins
    width: width of the histogram in characters
    """
    # Calculate histogram data
    hist, bin_edges = np.histogram(data, bins=bins)
    max_hist = max(hist)
    
    # Calculate scaling factor
    scale = width / max_hist if max_hist > 0 else 1
    
    # Print histogram
    print("\nHistogram:")
    print("-" * (width + 20))  # Header line
    
    for i in range(len(hist)):
        # Format bin range
        bin_start = f"{bin_edges[i]:.1f}"
        bin_end = f"{bin_edges[i+1]:.1f}"
        bin_range = f"[{bin_start}, {bin_end})"
        
        # Calculate bar length
        bar_length = int(hist[i] * scale)
        bar = "#" * bar_length
        
        # Print bar with count
        print(f"{bin_range:15} | {bar}{' ' * (width - bar_length)} | {hist[i]}")
    
    print("-" * (width + 20))  # Footer line
    print(f"Total count: {len(data)}")
    print(f"Mean: {np.mean(data):.2f}")
    print(f"Std dev: {np.std(data):.2f}")

def get_type_of_entry(entry):
    if entry.get('source', 'unknown') == 'mcq_26m' and entry['answer_type'] in ['a', 'b', 'c', 'd', 'e']:
        key = ('mcq_26m', entry['answer_type'])
    elif entry.get('source', 'unknown') == 'mcq_26m_close_form' and entry['answer_type'] in ['a', 'b', 'c']:
        key = ('mcq_26m_close_form', entry['answer_type'])
    else:
        key = (entry['Topic'], entry['answer_type'])
    return key

def process_single_response_key(item, output_key, response_key, enable_math_expr_extract, timeout=30):
    try:
        return run_with_timeout(
            process_entries,
            args=([item],),
            kwargs={
                'output_key': output_key,
                'model_output_key': response_key,
                'enable_math_expr_extract': enable_math_expr_extract,
                'use_ray': False
            },
            timeout=timeout
        )["correctness_rewards"][0]
    except TimeoutError:
        return 0.0
    except Exception as e:
        return 0.0

def process_item(item, output_key, multi_response_key, new_score_key, enable_math_expr_extract=False, timeout=60):
    """
    Process a single item with multiple response keys.
    
    Args:
        item: The item to process
        output_key: Key for output in the item
        multi_response_key: List of response keys to process
        new_score_key: Key for storing results
        enable_math_expr_extract: Whether to enable math expression extraction
        timeout: Timeout in seconds for processing each response key
        
    Returns:
        Tuple of (processed_item, None) if successful, (None, failed_item) if failed
    """
    try:
        results = []
        for response_key in multi_response_key:
            score = process_single_response_key(
                item,
                output_key,
                response_key,
                enable_math_expr_extract,
                timeout
            )
            results.append(score)
        
        if new_score_key not in item:
            item[new_score_key] = []
        item[new_score_key].extend(results)
        return item, None
    except Exception as e:
        print(f"Error processing item {item.get('uid', 'unknown')}: {str(e)}")
        return None, item

def process_file(file_path, compute_correctness=False, batch_size=10, show_progress=True, max_workers=10):
    """
    Process a single file and return uid to correct mapping and type distribution.
    Uses multiple processes to process the entire file at once.
    
    Args:
        file_path: Path to the input file
        compute_correctness: Whether to compute new correctness scores
        batch_size: Size of batches for processing (unused in this version)
        show_progress: Whether to show progress bar
        max_workers: Maximum number of processes to use
    """
    # Save new data to a different file
    output_dir = os.path.dirname(file_path)
    output_filename = os.path.basename(file_path).replace('.jsonl', '_with_correctness.jsonl')
    output_path = os.path.join(output_dir, output_filename)
    output_path_failed = os.path.join(output_dir, output_filename.replace('.jsonl', '_failed.jsonl'))
    
    new_score_key = "verifier-v3_correctness_score"
    multi_response_key = ['response'] + [f'response_{i}' for i in range(0, 8)]
    
    local_uid_to_correct = defaultdict(list)
    local_type_distribution = defaultdict(int)

    n_processed_lines = 0
    # load uid_to_correct from `output_path`
    if os.path.exists(output_path):
        finished_data = orjsonl.load(output_path)
        n_processed_lines += len(finished_data)
        for item in finished_data:
            local_uid_to_correct[item['uid']].extend(item[new_score_key])
            local_type_distribution[get_type_of_entry(item)] += 1
        del finished_data
    if os.path.exists(output_path_failed):
        failed_data = orjsonl.load(output_path_failed)
        n_processed_lines += len(failed_data)
        del failed_data
    
    data = orjsonl.load(file_path)
    print(f"Loaded {len(data)} lines from {file_path}")
    if n_processed_lines > 0:
        print(f"Found {n_processed_lines} lines in {output_path} and {output_path_failed}, skipping them")
        data = data[n_processed_lines:]
    print(f"About to process {len(data)} lines")
    
    start_time = time.time()

    if compute_correctness:
        # processed_items = []
        # failed_items = []
        
        # Create progress bar if requested
        if show_progress:
            pbar = tqdm(total=len(data), desc="Processing items")
        
        # Process items in chunks to better handle failures
        chunk_size = batch_size
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            chunk_processed = []
            chunk_failed = []
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit chunk of items
                future_to_item = {
                    executor.submit(
                        process_item,
                        item,
                        "answer",
                        multi_response_key,
                        new_score_key,
                        False,
                        10
                    ): item for item in chunk
                }
                
                # Process completed items
                # for future in as_completed(future_to_item, timeout=300):
                for future in as_completed(future_to_item):
                    try:
                        item, failed_item = future.result()  # 1 minutes timeout
                        if item is not None:
                            chunk_processed.append(item)
                        if failed_item is not None:
                            chunk_failed.append(failed_item)
                        if show_progress:
                            pbar.update(1)
                    except TimeoutError:
                        print(f"Timeout waiting for result from process")
                        chunk_failed.append(future_to_item[future])
                        if show_progress:
                            pbar.update(1)
                    except Exception as e:
                        print(f"Error in future result: {str(e)}")
                        chunk_failed.append(future_to_item[future])
                        if show_progress:
                            pbar.update(1)
            
            # Add processed items to final list
            # processed_items.extend(chunk_processed)
            # failed_items.extend(chunk_failed)
            
            # write to file
            tqdm.write(f"Writing {len(chunk_processed)} processed items to {output_path}")
            orjsonl.extend(output_path, chunk_processed)
            tqdm.write(f"Writing {len(chunk_failed)} failed items to {output_path_failed}")
            orjsonl.extend(output_path_failed, chunk_failed)
        
        if show_progress:
            pbar.close()
        
        end_time = time.time()
        print(f"\nProcessed {len(data)} entries in {end_time - start_time:.2f} seconds")
        # print(f"Successfully processed: {len(processed_items)}")
        # print(f"Failed to process: {len(failed_items)}")
        

        # check the number of lines in `output_path`
        # num_lines = sum(1 for _ in open(output_path))
        # print(f"Number of lines in {output_path}: {num_lines}, should be {len(processed_items)}")
        
        # start_time = time.time()
        # orjsonl.save(output_path, processed_items)
        # end_time = time.time()
        # print(f"Saved new data to {output_path} in {end_time - start_time:.2f} seconds")
        
        # # Save failed items to a separate file
        # if failed_items:
        #     failed_filename = os.path.basename(file_path).replace('.jsonl', '_failed.jsonl')
        #     failed_path = os.path.join(output_dir, failed_filename)
        #     orjsonl.save(failed_path, failed_items)
        #     print(f"Saved failed items to {failed_path}")
        
        # data = processed_items
    
    # Process rewards
    data = orjsonl.load(output_path)
    for item in data:
        if compute_correctness or '_with_correctness.jsonl' in file_path:
            if isinstance(item[new_score_key], list):
                local_uid_to_correct[item['uid']].extend(item[new_score_key])
            else:
                local_uid_to_correct[item['uid']].append(item[new_score_key])
        entry_type = get_type_of_entry(item)
        local_type_distribution[entry_type] += 1
    
    return dict(local_uid_to_correct), dict(local_type_distribution)


def merge_results(results):
    """Merge results from multiple processes."""
    merged_scores = defaultdict(list)
    merged_types = defaultdict(int)
    
    for result_scores, result_types in results:
        for uid, scores in result_scores.items():
            merged_scores[uid].extend(scores)
        for entry_type, count in result_types.items():
            merged_types[entry_type] += count
            
    return merged_scores, merged_types

def print_type_distribution(type_distribution):
    """Print the distribution of entry types in a formatted way."""
    print("\nEntry Type Distribution:")
    print("-" * 80)
    print(f"{'Entry Type':50} | {'Count':10} | {'Percentage':10}")
    print("-" * 80)
    
    total = sum(type_distribution.values())
    sorted_types = sorted(type_distribution.items(), key=lambda x: x[1], reverse=True)
    
    for entry_type, count in sorted_types:
        source, answer_type = entry_type
        type_str = f"{source} - {answer_type}"
        percentage = (count / total) * 100
        print(f"{type_str:50} | {count:10d} | {percentage:8.2f}%")
    
    print("-" * 80)
    print(f"Total entries: {total}")


def main(
    local_dir = 'inference_results/infer_result_gen1/infer_result_splitted',
    file_start=-1, 
    file_end=-1,
    compute_correctness=True,
    max_workers_per_batch=128,
    batch_size=2000, # 每个文件的 batch size，这个太大会卡住
    ):

    # Get all files
    all_file_list = glob(os.path.join(local_dir, '*.jsonl'))
    print('Total files found:', len(all_file_list))
    
    if compute_correctness:
        all_file_list = sorted(all_file_list)
        # Filter out files that already have correctness scores
        all_file_list = [p for p in all_file_list if '_with_correctness' not in p]
        if file_start >= 0 and file_end >= 0:
            all_file_list = all_file_list[file_start:file_end]
            
        print('Files to process:', len(all_file_list))
    else:
        # Only process files with correctness scores
        all_file_list = [p for p in all_file_list if '_with_correctness.jsonl' in p]
        print('Files with correctness scores:', len(all_file_list))

    
    if not all_file_list:
        print("No files to process!")
        return

    # Process files sequentially
    uid_to_correct = defaultdict(list)
    type_distribution = defaultdict(int)
    
    for file_path in tqdm(all_file_list, desc="Processing files"):
        try:
            # Process each file
            result_scores, result_types = process_file(
                file_path,
                compute_correctness=compute_correctness,
                batch_size=batch_size,
                show_progress=True,
                max_workers=max_workers_per_batch
            )
            
            # Merge results
            for uid, scores in result_scores.items():
                uid_to_correct[uid].extend(scores)
            for entry_type, count in result_types.items():
                type_distribution[entry_type] += count
                
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            continue
    
    if not compute_correctness:
        # Print type distribution
        print_type_distribution(type_distribution)
        
        # Plot distributions
        print("\nDistribution of number of entries per UID:")
        create_histogram_ascii([len(v) for v in uid_to_correct.values()], bins=5, width=30)
        
        print("\nDistribution of total scores per UID:")
        create_histogram_ascii([sum(v) for v in uid_to_correct.values()], bins=64, width=30)
        
        print(f"\nNumber of UIDs with perfect scores: {len([uid for uid in uid_to_correct if 1.0 in uid_to_correct[uid]])}")

        # Save results
        output_path = os.path.join(local_dir, 'uid_to_correct.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(uid_to_correct, f)
        print(f"\nResults saved to {output_path}")
    else:
        print('Finished computing correctness scores. Set compute_correctness=False to analyze results.')

if __name__ == '__main__':
    from fire import Fire
    Fire(main)
