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

def _worker(job):
    start, end, tokenizer, file_path = job
    with open(file_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(end-start).decode('utf-8', errors='ignore')
    return tokenizer.encode(chunk)

def tokenize_text_to_token_ids_and_save(
    file_path: str,
    tokenizer: BPETokenizer,
    output_file_path: str | Path,
    show_prograss=False
):
    assert os.path.exists(file_path), print(f'file_path ({file_path}) cannot be found!')

    num_processes=64

    with Path(file_path).open('rb') as f:
        boundaries = find_chunk_boundaries(f, num_processes, b'<|endoftext|>')
    
    boundaries = list(zip(boundaries[:-1], boundaries[1:]))
    results = []

    chunk_per_worker = (len(boundaries) + num_processes - 1) // num_processes
    jobs = [
        (
            start,
            end, 
            tokenizer, 
            file_path
        ) for start, end in boundaries
    ]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_processes) as pool:
        with tqdm(total=len(jobs), desc='Tokenizing', unit='chunk') as progress:
            for res in pool.imap(_worker, jobs):
                results += res
            progress.update(1)

    # save the token_ids to a file 
    results = np.array(results, dtype=np.uint16)
    np.save(output_file_path, results)

def tokenize_owt():
    # hardcode the path to owt_train, owt_valid, merges, and vocab
    owt_train_path = '/global/cfs/cdirs/m4410/mzheng/cs336/assignment1-basics/data/owt_train.txt'
    owt_valid_path = '/global/cfs/cdirs/m4410/mzheng/cs336/assignment1-basics/data/owt_valid.txt'
    owt_vocab_path = '/global/cfs/cdirs/m4410/mzheng/cs336/assignment1-basics/data/owt_train_vocab.json'
    owt_merges_path = '/global/cfs/cdirs/m4410/mzheng/cs336/assignment1-basics/data/owt_train_merges.txt'

    tokenizer = BPETokenizer.from_files(
        vocab_file_path=owt_vocab_path,
        merges_file_path=owt_merges_path,
        special_tokens=['<|endoftext|>']
    )

    owt_train_tokenized_output_file_path = '/global/cfs/cdirs/m4410/mzheng/cs336/assignment1-basics/data/owt_train.npy'
    print("start tokenizing owt.train")
    tokenize_text_to_token_ids_and_save(
        file_path=owt_train_path,
        tokenizer=tokenizer,
        output_file_path=owt_train_tokenized_output_file_path,
        show_prograss=True
    )
    print(f"tokenized owt_train.txt saved to {owt_train_tokenized_output_file_path}")

    owt_valid_tokenized_output_file_path = '/global/cfs/cdirs/m4410/mzheng/cs336/assignment1-basics/data/owt_valid.npy'
    print("start tokenizing owt.valid")
    tokenize_text_to_token_ids_and_save(
        file_path=owt_valid_path,
        tokenizer=tokenizer,
        output_file_path=owt_valid_tokenized_output_file_path,
        show_prograss=True
    )
    print(f"tokenized owt_valid.txt saved to {owt_train_tokenized_output_file_path}")


if __name__ == '__main__':
    tokenize_owt()