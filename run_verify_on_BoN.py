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
from local_verifier import process_entries

# import ray

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

def process_file(file_path, compute_correctness=False, batch_size=10, show_progress=True):
    """
    Process a single file and return uid to correct mapping and type distribution.
    
    Args:
        file_path: Path to the input file
        compute_correctness: Whether to compute new correctness scores
        batch_size: Size of batches for processing
        show_progress: Whether to show progress bar
    """
    local_uid_to_correct = defaultdict(list)
    local_type_distribution = defaultdict(int)
    data = orjsonl.load(file_path) # [:10] # test
    start_time = time.time()
    new_score_key = "verifier-v3_correctness_score"
    multi_response_key = ['response'] + [f'response_{i}' for i in range(0, 8)]
    
    if compute_correctness:
        new_data = []
        total_batches = (len(data) + batch_size - 1) // batch_size
        
        # Create progress bar if requested
        batch_iterator = range(0, len(data), batch_size)
        if show_progress:
            batch_iterator = tqdm(batch_iterator, total=total_batches, desc="Processing batches")
        
        for start_idx in batch_iterator:
            end_idx = min(start_idx + batch_size, len(data))
            batch = data[start_idx:end_idx]

            # Create new entries with correctness scores for this batch
            new_data_batch = deepcopy(batch)
                
            for response_key in multi_response_key:
                # Process batch
                batch_results = process_entries(
                    entries=new_data_batch,
                    output_key="answer",
                    model_output_key=response_key,
                    enable_math_expr_extract=False,
                    use_ray=False
                    # use_ray=True,
                )
                results = batch_results["correctness_rewards"]
                # save batch_scores to the new items in new_data
                for idx, (item, result) in enumerate(zip(new_data_batch, results)):
                    if new_score_key not in item:
                        item[new_score_key] = []
                    # item[new_score_key].append((response_key, result["correctness_rewards"][idx]))
                    item[new_score_key].append(result)
            
            new_data.extend(new_data_batch)
        end_time = time.time()
        print(f"\nProcessed {len(data)} entries in {end_time - start_time:.2f} seconds")
        
        # Save new data to a different file
        output_dir = os.path.dirname(file_path)
        output_filename = os.path.basename(file_path).replace('.jsonl', '_with_correctness.jsonl')
        output_path = os.path.join(output_dir, output_filename)
        
        start_time = time.time()
        orjsonl.save(output_path, new_data)
        end_time = time.time()
        print(f"Saved new data to {output_path} in {end_time - start_time:.2f} seconds")
        data = new_data  # Use new data for subsequent processing
    
    # Process rewards
    for item in data:
        if compute_correctness or '_with_correctness.jsonl' in file_path:
            if isinstance(item[new_score_key], list):
                local_uid_to_correct[item['uid']].extend(item[new_score_key])
            else:
                local_uid_to_correct[item['uid']].append(item[new_score_key])
        # else:
        #     local_uid_to_correct[item['uid']].append(item["llama-sft2_7b_Bo64_output_reward"])
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
        with mp.Pool(num_processes) as pool:
            process_file_with_correctness = partial(process_file, compute_correctness=compute_correctness)
            results = list(tqdm(
                pool.imap(process_file_with_correctness, file_list),
                total=len(file_list),
                desc=f"Processing total {len(file_list)} files"
            ))
        uid_to_correct, type_distribution = merge_results(results)
    else:
        print("Using single process")
        uid_to_correct = defaultdict(list)
        type_distribution = defaultdict(int)
        for f in tqdm(file_list, desc="Processing files"):
            result_scores, result_types = process_file(f, compute_correctness)
            for uid, scores in result_scores.items():
                uid_to_correct[uid].extend(scores)
            for entry_type, count in result_types.items():
                type_distribution[entry_type] += count
    
    return uid_to_correct, type_distribution

def main(file_start=0, file_end=0):
    # ray.init() 
    # ray.init(address='auto')
    # Command line arguments could be added here
    use_multiprocessing = True # False
    num_processes = 96 # mp.cpu_count() - 4  # 64
    compute_correctness = True  # New option to compute correctness scores, set False to use existing scores but not do inferencing
    # compute_correctness = False
    
    local_dir = 'inference_results/infer_result_gen1/infer_result_splitted'
    
    # Get all files
    all_file_list = glob(os.path.join(local_dir, '*.jsonl'))
    print('get all files:', len(all_file_list))
    
    if compute_correctness:
        all_finished_file_list = [p for p in all_file_list if '_with_correctness.jsonl' not in p]
        print('get finished files:', len(all_finished_file_list))
        # all_file_list = [p for p in all_file_list if '_with_correctness.jsonl' not in p]
        tmp_list = []
        for p in all_file_list:
            if '_with_correctness.jsonl' not in p and p.replace('.jsonl', '_with_correctness.jsonl') not in all_finished_file_list:
                tmp_list.append(p)
        all_file_list = tmp_list
        print('remained files:', len(all_file_list))
    else:
        # compute correctness=False, only process files with correctness
        all_file_list = [p for p in all_file_list if '_with_correctness.jsonl' in p]    

    all_file_list = sorted(all_file_list)
    if file_start >= 0 and file_end > 0:
        all_file_list = all_file_list[file_start:file_end]
    
    # print('total file:', len(all_file_list))
    # exit(0)
    
    num_processes = min(num_processes, len(all_file_list))
    # Process files
    uid_to_correct, type_distribution = process_files(
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