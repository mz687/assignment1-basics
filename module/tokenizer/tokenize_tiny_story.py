import os
import regex as re
import multiprocessing as mp
from .common import gpt2_bytes_to_unicode
import json
from pathlib import Path
from collections.abc import Iterable, Iterator
import heapq
from .tokenizer import find_chunk_boundaries
import numpy as np
from tqdm import tqdm
from .BPETokenizer import BPETokenizer

def tokenize_text_to_token_ids_and_save(
    file_path: str,
    tokenizer: BPETokenizer,
    output_file_path: str | Path,
    show_prograss=False
):
    assert os.path.exists(file_path), print(f'file_path ({file_path}) cannot be found!')

    with Path(file_path).open('rb') as f:
        num_processes=1
        boundaries = find_chunk_boundaries(f, num_processes, b'<|endoftext|>')

    results = []
    # serialized implementation. should be implemented by mp
    for start, end in boundaries:
        with open(file_path, 'rb') as f:
            f.seek(start)
            chunk = f.read(end-start).decode('utf-8', errors='ignore')
            ids = tokenizer.encode(chunk, show_prograss)
            results += ids
    
    results = np.array(results, dtype=np.uint16)
    np.save(output_file_path, results)

def tokenize_tinystories():
    tiny_story_train_path='/global/cfs/cdirs/m4410/mzheng/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt'
    tiny_story_valid_path='/global/cfs/cdirs/m4410/mzheng/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt'
    tiny_story_vocab_path = '/global/cfs/cdirs/m4410/mzheng/cs336/assignment1-basics/data/TinyStoriesV2-GPT4_vocab.json'
    tiny_story_merges_path = '/global/cfs/cdirs/m4410/mzheng/cs336/assignment1-basics/data/TinyStoriesV2-GPT4_merges.txt'

    tokenizer = BPETokenizer.from_files(
        vocab_file_path=tiny_story_vocab_path,
        merges_file_path=tiny_story_merges_path,
        special_tokens=['<|endoftext|>']
    )

    print("start tokenizing tiny_story.train")
    tokenize_text_to_token_ids_and_save(
        file_path=tiny_story_train_path,
        tokenizer=tokenizer,
        output_file_path='/global/cfs/cdirs/m4410/mzheng/cs336/assignment1-basics/data/tinystories_train.pkl',
        show_prograss=True
    )

    print('start tokenizing tiny_story.valid')
    tokenize_text_to_token_ids_and_save(
        file_path=tiny_story_valid_path,
        tokenizer=tokenizer,
        output_file_path='/global/cfs/cdirs/m4410/mzheng/cs336/assignment1-basics/data/tinystories_valid.pkl',
        show_prograss=True
    )


if __name__ == '__main__':
    tokenize_tinystories()