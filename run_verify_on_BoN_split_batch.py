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
import signal
from local_verifier import process_entries
import psutil
import traceback
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
# import ray

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Process timeout")

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
    if entry['source'] == 'mcq_26m' and entry['answer_type'] in ['a', 'b', 'c', 'd', 'e']:
        key = ('mcq_26m', entry['answer_type'])
    elif entry['source'] == 'mcq_26m_close_form' and entry['answer_type'] in ['a', 'b', 'c']:
        key = ('mcq_26m_close_form', entry['answer_type'])
    else:
        key = (entry['Topic'], entry['answer_type'])
    return key

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def process_items(items, compute_correctness=False):
    """
    Process a list of items and return uid to correct mapping and type distribution.
    
    Args:
        items: List of items to process
        compute_correctness: Whether to compute new correctness scores
    """
    local_uid_to_correct = defaultdict(list)
    local_type_distribution = defaultdict(int)
    new_score_key = "verifier-v3_correctness_score"
    multi_response_key = ['response'] + [f'response_{i}' for i in range(0, 8)]
    
    if compute_correctness:
        # Process all responses in one pass to avoid multiple deep copies
        new_items = deepcopy(items)
        
        for response_key in multi_response_key:
            try:
                # Process batch
                batch_results = process_entries(
                    entries=new_items,
                    output_key="answer", 
                    model_output_key=response_key,
                    enable_math_expr_extract=False,
                    use_ray=False
                )
                results = batch_results["correctness_rewards"]
                
                # Save scores to the items
                for idx, (item, result) in enumerate(zip(new_items, results)):
                    if new_score_key not in item:
                        item[new_score_key] = []
                    item[new_score_key].append(result)
                    
            except Exception as e:
                print(f"ERROR: Error processing response {response_key} for batch:")
                print(f"ERROR: Exception type: {type(e).__name__}")
                print(f"ERROR: Exception message: {str(e)}")
                print(f"ERROR: Traceback: {traceback.format_exc()}")
                # Set all scores to 0 for this response key
                for item in new_items:
                    if new_score_key not in item:
                        item[new_score_key] = []
                    item[new_score_key].append(0.0)
        
        items = new_items
    
    # Process rewards and types in a single pass
    for item in items:
        if compute_correctness:
            if isinstance(item[new_score_key], list):
                local_uid_to_correct[item['uid']].extend(item[new_score_key])
            else:
                local_uid_to_correct[item['uid']].append(item[new_score_key])
        entry_type = get_type_of_entry(item)
        local_type_distribution[entry_type] += 1
    
    return dict(local_uid_to_correct), dict(local_type_distribution), items

def merge_results(results):
    """Merge results from multiple processes."""
    merged_scores = defaultdict(list)
    merged_types = defaultdict(int)
    processed_items = []
    
    for result in results:
        # Handle both old format (scores, types) and new format (scores, types, items)
        if len(result) == 2:
            result_scores, result_types = result
        else:
            result_scores, result_types, items = result
            processed_items.extend(items)
            
        for uid, scores in result_scores.items():
            merged_scores[uid].extend(scores)
        for entry_type, count in result_types.items():
            merged_types[entry_type] += count
            
    return merged_scores, merged_types, processed_items

def process_file_with_items(file_path, num_processes, compute_correctness=False):
    """
    Process a file by splitting its items into batches for parallel processing.
    
    Args:
        file_path: Path to the input file
        num_processes: Number of processes to use
        compute_correctness: Whether to compute new correctness scores
    """
    print(f"\nDEBUG: Starting to process file: {file_path}")
    
    # Load all items from the file
    items = orjsonl.load(file_path)
    print(f"DEBUG: Loaded {len(items)} items from {file_path}")
    
    # Split items into larger batches for better parallelization
    batch_size = max(50, len(items) // (num_processes * 4))  # Ensure at least 50 items per batch
    item_batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    print(f"DEBUG: Split into {len(item_batches)} batches of size {batch_size}")
    
    # Process batches in parallel using ProcessPoolExecutor
    print("DEBUG: Starting parallel processing with multiprocessing")
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        process_items_with_correctness = partial(process_items, compute_correctness=compute_correctness)
        pbar = tqdm(
            total=len(item_batches),
            desc=f"Processing {len(items)} items from {os.path.basename(file_path)}",
            position=0,
            leave=True
        )
        results = []
        futures = [executor.submit(process_items_with_correctness, batch) for batch in item_batches]
        for future in as_completed(futures):
            results.append(future.result())
            pbar.update(1)
        pbar.close()
    
    print("DEBUG: Finished processing all batches, merging results...")
    # Merge results from all batches
    uid_to_correct, type_distribution, processed_items = merge_results(results)
    print("DEBUG: Successfully merged results")
    
    # If computing correctness, save the processed items
    if compute_correctness:
        output_dir = os.path.dirname(file_path)
        output_filename = os.path.basename(file_path).replace('.jsonl', '_with_correctness.jsonl')
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"DEBUG: Saving processed items to {output_path}")
        start_time = time.time()
        orjsonl.save(output_path, processed_items)
        end_time = time.time()
        print(f"DEBUG: Saved {len(processed_items)} items to {output_path} in {end_time - start_time:.2f} seconds")
    
    return uid_to_correct, type_distribution, processed_items

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

def process_files(file_list, use_multiprocessing=False, num_processes=None, compute_correctness=False):
    """
    Process files either using single process or multiprocessing.
    
    Parameters:
    file_list: list of files to process
    use_multiprocessing: whether to use multiprocessing
    num_processes: number of processes to use (defaults to CPU count if None)
    compute_correctness: whether to compute and save correctness scores
    """
    if use_multiprocessing:
        if num_processes is None:
            num_processes = mp.cpu_count()
        
        print(f"Using multiprocessing with {num_processes} processes")
        process_file_with_items_with_correctness = partial(process_file_with_items, num_processes=num_processes, compute_correctness=compute_correctness)
        results = list(tqdm(
            map(process_file_with_items_with_correctness, file_list),
            total=len(file_list),
            desc=f"Processing total {len(file_list)} files"
        ))
        uid_to_correct, type_distribution, processed_items = merge_results(results)
    else:
        print("Using single process")
        uid_to_correct = defaultdict(list)
        type_distribution = defaultdict(int)
        for f in tqdm(file_list, desc="Processing files"):
            result_scores, result_types, processed_items = process_file_with_items(f, num_processes=1, compute_correctness=compute_correctness)
            for uid, scores in result_scores.items():
                uid_to_correct[uid].extend(scores)
            for entry_type, count in result_types.items():
                type_distribution[entry_type] += count
    
    return uid_to_correct, type_distribution, processed_items

def main(file_start=-1, file_end=-1):
    # ray.init() 
    # ray.init(address='auto')
    # Command line arguments could be added here
    use_multiprocessing = True # False
    num_processes = 80 # mp.cpu_count() - 4  # 64
    compute_correctness = True  # New option to compute correctness scores, set False to use existing scores but not do inferencing
    # compute_correctness = False
    
    # local_dir = 'inference_results/infer_result_gen1/infer_result'
    local_dir = 'debug'
    
    # Get all files
    all_file_list = glob(os.path.join(local_dir, '*.jsonl'))
    print('get all files:', len(all_file_list))
    
    # if compute_correctness:
    #     all_finished_file_list = [p for p in all_file_list if '_with_correctness.jsonl' in p]
    #     print('get finished files:', len(all_finished_file_list))
    #     # all_file_list = [p for p in all_file_list if '_with_correctness.jsonl' not in p]
    #     tmp_list = []
    #     for p in all_file_list:
    #         if '_with_correctness.jsonl' not in p and p.replace('.jsonl', '_with_correctness.jsonl') not in all_finished_file_list:
    #             tmp_list.append(p)
    #     all_file_list = tmp_list
    #     print('remained files:', len(all_file_list))
    # else:
    #     # compute correctness=False, only process files with correctness
    #     all_file_list = [p for p in all_file_list if '_with_correctness.jsonl' in p]    

    all_file_list = sorted(all_file_list)
    if file_end != -1 and file_start != -1:
        all_file_list = all_file_list[file_start:file_end]
    print('keep files:', len(all_file_list))
    # print('total file:', len(all_file_list))
    # exit(0)
    
    num_processes = min(num_processes, mp.cpu_count()-4)
    # Process files
    uid_to_correct, type_distribution, processed_items = process_files(
        all_file_list, 
        use_multiprocessing, 
        num_processes,
        compute_correctness
    )
    
    if compute_correctness:
        print('finish, now set compute_correctness=False to run again')
        exit()
    
    # Print type distribution
    print_type_distribution(type_distribution)
    
    # Plot distributions
    print("\nDistribution of number of entries per UID:")
    create_histogram_ascii([len(v) for v in uid_to_correct.values()], bins=5, width=30)
    
    # print("\nDistribution of average scores per UID:")
    # create_histogram_ascii([sum(v)/len(v) for v in uid_to_correct.values()], bins=3, width=30)
    print("\nDistribution of total scores per UID:")
    create_histogram_ascii([sum(v) for v in uid_to_correct.values()], bins=64, width=30)
    
    
    print(f"\nNumber of UIDs with perfect scores: {len([uid for uid in uid_to_correct if 1.0 in uid_to_correct[uid]])}")

    # Save results
    output_path = os.path.join(local_dir, 'uid_to_correct.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(uid_to_correct, f)
    print(f"\nResults saved to {output_path}")

    # ray.close()

if __name__ == '__main__':
    from fire import Fire
    Fire(main)