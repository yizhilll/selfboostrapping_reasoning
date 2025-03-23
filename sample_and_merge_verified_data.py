import orjsonl
from glob import glob
import os
from collections import defaultdict
import random
from typing import Dict, List, Optional, Tuple
import multiprocessing as mp
from tqdm import tqdm
import time
from copy import deepcopy
import numpy as np

def find_output_tags(text: str) -> Optional[Tuple[int, int, int, int]]:
    """
    Find the start and end positions of Output tags and their content.
    
    Returns:
        Tuple of (start_tag_start, start_tag_end, end_tag_start, end_tag_end)
        or None if tags not found or malformed
    """
    if not isinstance(text, str):
        return None

    start_tag = "<Output>"
    end_tag = "</Output>"
    
    start_tag_start = text.find(start_tag)
    if start_tag_start == -1:
        return None
        
    start_tag_end = start_tag_start + len(start_tag)
    end_tag_start = text.find(end_tag, start_tag_end)
    if end_tag_start == -1:
        return None
        
    # Verify there's no nested Output tag
    next_start_tag = text.find(start_tag, start_tag_end)
    if next_start_tag != -1 and next_start_tag < end_tag_start:
        return None
        
    end_tag_end = end_tag_start + len(end_tag)
    return (start_tag_start, start_tag_end, end_tag_start, end_tag_end)

def is_item_valid(item: Dict) -> bool:
    """Check if an item has all required fields and valid format."""
    if not all(key in item for key in ['infer_bon64', 'boxed', 'verifier-v3_correctness_score']):
        return False
    
    if not isinstance(item['verifier-v3_correctness_score'], (int, float)):
        return False
    if item['verifier-v3_correctness_score'] <= 0:
        return False
    
    if not find_output_tags(item['infer_bon64']):
        return False
        
    return True

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

def replace_output_content(infer_bon64: str, boxed_content: str) -> Optional[str]:
    """Replace content between <Output> and </Output> tags with new content."""
    tag_positions = find_output_tags(infer_bon64)
    if not tag_positions:
        return None
        
    start_tag_start, start_tag_end, end_tag_start, end_tag_end = tag_positions
    
    boxed_content = boxed_content.replace('\x08oxed', '\\boxed')
    try:
        new_output = (
            infer_bon64[:start_tag_end] + 
            "\n" + boxed_content + "\n" +
            infer_bon64[end_tag_start:]
        )
        return new_output
    except Exception:
        return None

def clean_item(item: Dict) -> Dict:
    """Remove specified columns from item."""
    columns_to_remove = ['infer_bon64', 'llama-sft2_7b_Bo64_output_reward', 'boxed']
    return {k: v for k, v in item.items() if k not in columns_to_remove}

def process_single_file(mp_args) -> Tuple[List[Dict], List[Dict]]:
    """
    Process a single file and return both valid items and sampled items.
    
    Returns:
        Tuple of (all valid items, sampled items)
    """
    # file_path: str, sample_n: int = 1
    file_path, sample_n = mp_args
    try:
        start_time = time.time()
        uid_to_items = defaultdict(list)
        data = orjsonl.load(file_path)
        load_time = time.time() - start_time
        if random.random() < 0.01:  # Log timing for ~1% of files
            print(f"File loading time for {os.path.basename(file_path)}: {load_time:.2f} seconds")
        
        all_valid_items = []
        # sampled_items = []
        
        for item in data:
            if 'uid' not in item:
                continue
            uid_to_items[item['uid']].append(item)
        
        for uid, items in uid_to_items.items():
            valid_items = []
            for item in items:
                if not is_item_valid(item):
                    continue
                    
                new_output = replace_output_content(item['infer_bon64'], item['boxed'])
                if new_output is None:
                    continue
                is_legal_split, _ = is_legal_cot_split_boxed(new_output)
                if not is_legal_split:
                # if is_legal_split:
                    continue
                    
                item_copy = deepcopy(item)
                item_copy['boxed_thought_output'] = new_output
                # item_copy = clean_item(item_copy)
                valid_items.append(item_copy)
            
            if valid_items:
                all_valid_items.extend(valid_items)
                # sampled_items.append(random.choice(valid_items))
                # sampled_items.extend(np.random.choice(valid_items, min(sample_n, len(valid_items)), replace=False).tolist())
                
        return all_valid_items
        # return all_valid_items, sampled_items
        
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        # return [], []
        return []

def merge_and_sample(
    input_pattern: str,
    base_output_path: str,
    num_processes: int = None,
    chunk_size: int = 10,
    n_sample: int = 1,
) -> None:
    """Merge and sample from multiple JSONL files."""
    start_time = time.time()
    all_files = glob(input_pattern)
    random.shuffle(all_files)
    # all_files = all_files[:5] # debug
    if not all_files:
        print(f"No files found matching pattern: {input_pattern}")
        return
        
    print(f"Found {len(all_files)} files to process")
    
    if num_processes is None:
        num_processes = mp.cpu_count() -4 
    
    mp_args = [(fp, n_sample) for fp in all_files]
    
    with mp.Pool(num_processes) as pool:
        all_valid_items = []
        # sampled_items = []
        
        with tqdm(total=len(all_files), desc="Processing files") as pbar:
            for valid_items in pool.imap_unordered(process_single_file, mp_args, chunksize=chunk_size):
                all_valid_items.extend(valid_items)
                # sampled_items.extend(sampled)
                pbar.update()
    
    print(f"\nProcessing summary:")
    print(f"Total files processed: {len(all_files)}")
    print(f"Total valid items: {len(all_valid_items)}")
    # print(f"Total sampled items: {len(sampled_items)}")
    
    if not all_valid_items:
        print("No valid items found!")
        return
    random.shuffle(all_valid_items)
    # Save all valid items in chunks
    CHUNK_SIZE = 2_000_000
    total_items = len(all_valid_items)
    num_chunks = (total_items + CHUNK_SIZE - 1) // CHUNK_SIZE  # Ceiling division
    
    total_save_start = time.time()
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * CHUNK_SIZE
        end_idx = min((chunk_idx + 1) * CHUNK_SIZE, total_items)
        chunk_items = all_valid_items[start_idx:end_idx]
        
        # Generate filename for this chunk
        if num_chunks > 1:
            chunk_path = base_output_path.format(number_of_items=f"{total_items}_part{chunk_idx+1}of{num_chunks}")
        else:
            chunk_path = base_output_path.format(number_of_items=total_items)
        
        print(f"\nSaving chunk {chunk_idx + 1}/{num_chunks} ({len(chunk_items)} items) to: {chunk_path}")
        save_start = time.time()
        orjsonl.save(chunk_path, chunk_items, compression_threads=64)
        save_time = time.time() - save_start
        print(f"Saved chunk in {save_time:.2f} seconds")
        print(f"Chunk save speed: {len(chunk_items)/save_time:.2f} items/second")
    
    total_save_time = time.time() - total_save_start
    print(f"\nTotal saving time: {total_save_time:.2f} seconds")
    print(f"Overall save speed: {total_items/total_save_time:.2f} items/second")
    
    uid_to_items = defaultdict(list)
    for item in all_valid_items:
        uid_to_items[item['uid']].append(item)
        
    for n_sample in tqdm(range(1, 64), desc='Sampling'):
        # Sample items
        sampled_items = []
        for uid, items in uid_to_items.items():
            sampled_items.extend(np.random.choice(items, min(n_sample, len(items)), replace=False).tolist())
        
        # Save sampled items
        sampled_path = base_output_path.replace('.jsonl', f'_sampled-{n_sample}.jsonl').format(
            number_of_items=len(sampled_items)
        )
        print(f"\nSaving sampled items to: {sampled_path}")
        sample_save_start = time.time()
        random.shuffle(sampled_items)
        orjsonl.save(sampled_path, sampled_items, compression_threads=64)
        sample_save_time = time.time() - sample_save_start
        print(f"Saved {len(sampled_items)} sampled items in {sample_save_time:.2f} seconds")
        print(f"Average save speed: {len(sampled_items)/sample_save_time:.2f} items/second")
        
    # Print total execution time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

def main():
    # Configuration
    input_pattern = '/aifs4su/mmcode/codeclm/o1/OpenO1_SFT_ultra_BoN_rewarded/*/*_with_correctness.jsonl'
    # base_output_path = '/aifs4su/mmcode/codeclm/o1/OpenO1_SFT_ultra_BoN_positvie_reward_v3_debug/math_merge_n-{number_of_items}.jsonl.gz'
    # base_output_path = '/aifs4su/mmcode/codeclm/o1/OpenO1_SFT_ultra_BoN_positvie_reward_v3_debug/math_merge_n-{number_of_items}.jsonl'
    # base_output_path = '/aifs4su/mmcode/codeclm/o1/OpenO1_SFT_ultra_BoN_positvie_reward_v3_debug_keep_illegal_split/math_merge_n-{number_of_items}.jsonl'
    base_output_path = '/aifs4su/mmcode/codeclm/o1/OpenO1_SFT_ultra_BoN_positvie_reward_v3_N-sample/math_merge_n-{number_of_items}.jsonl.gz'
    num_processes = 128 # None # 64
    chunk_size = 5
    
    os.makedirs(os.path.dirname(base_output_path), exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Run merge and sample
    # for n_sample in range(1, 64):
    merge_and_sample(
        input_pattern=input_pattern,
        base_output_path=base_output_path,
        num_processes=num_processes,
        chunk_size=chunk_size,
        n_sample=1,
    )

if __name__ == '__main__':
    main()