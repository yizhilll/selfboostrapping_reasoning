import json
import sys
from typing import Optional

def peek_jsonl(file_path: str) -> Optional[dict]:
    """
    Read the first line of a JSONL file and return it as a dictionary.
    
    Args:
        file_path (str): Path to the JSONL file
        
    Returns:
        Optional[dict]: First line as dictionary or None if file is empty
    """
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            if not first_line:
                print(f"File {file_path} is empty")
                return None
            return json.loads(first_line)
    except FileNotFoundError:
        print(f"File {file_path} not found")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON in {file_path}")
        return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python peek_jsonl.py <path_to_jsonl_file>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    data = peek_jsonl(file_path)
    
    if data:
        print("\nKeys in the JSON object:")
        for key in data.keys():
            print(f"- {key}")
            
        print("\nSample line (formatted):")
        print(json.dumps(data, indent=2))

if __name__ == "__main__":
    main()
