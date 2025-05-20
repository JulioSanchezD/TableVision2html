import json
import signal
from typing import List
from pathlib import Path


def append_json_line(file_path: str, data: dict):
    """
    Appends a dictionary as a new JSON line in a JSONL file.
    If the file doesn't exist, it creates it.

    :param file_path: Path to the JSONL file.
    :param data: Dictionary to append as a new JSON line.
    """
    # Ensure the data is a dictionary
    if not isinstance(data, dict):
        raise ValueError("Data must be a dictionary.")

    # Create parent directories if they don't exist
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    # Open the file in append mode and write the dictionary as a JSON line
    with open(file_path, 'a', encoding='utf-8') as file:
        json_line = json.dumps(data, ensure_ascii=False)
        file.write(json_line + '\n')


def read_jsonl(file_path: str) -> List[dict]:
    """
    Reads a JSON Lines file and returns a list of dictionaries.

    :param file_path: The path to the JSON Lines file.
    :return: list of dictionaries parsed from the file.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {line.strip()}")
                raise Exception(f"Error decoding JSON line: {str(e)}")
    return data

class TimeoutException(Exception):
    pass

def timeout(seconds=10):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutException(f"Function call timed out after {seconds} seconds")
        
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)  # Disable alarm
        return wrapper
    return decorator
