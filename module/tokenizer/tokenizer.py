import os
import multiprocessing as mp 
import regex as re
from typing import Dict, BinaryIO
from collections import defaultdict, Counter
from tqdm import tqdm
import time
import heapq
import tracemalloc

TEST_TEXT = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def init_vocab(vocab_table: Dict[bytes, int]):
    '''
    initialize the vocabulary table with 256 byte values and <|endoftext|> token
    remember: 
    str.encode('utf-8') converts string to bytes.
    bytes().decode('utf-8') converts bytes to string-ish.
    '''
    vocab_table[b'<|endoftext|>'] = 0
    for i in range(256):
        vocab_table[bytes([i])] = i + 1

def pretokenize(text: str) -> Dict[tuple[bytes, ...], int]:
    match_freq_table = defaultdict(int) # Dict[tuple[bytes, ...], int]
    for match in re.finditer(PAT, text):
        content = match.group()
        match_freq_table[content] += 1
    return match_freq_table

def reverse_lex_key(token: bytes):
    return tuple(-byte for byte in token) + (1,)

def entry(pair, count, token_rank):
    left_id, right_id = pair
    return (
        -count,
        token_rank[left_id],
        token_rank[right_id],
        left_id,
        right_id
    )

def merge(
    vocab_table: Dict[bytes, int],
    match_freq_table: Dict[tuple[bytes, ...], int], 
    iterations: int,
    show_prograss: bool=False
):
    reverse_vocab_table = {val: key for key, val in vocab_table.items()}

    # word_tokens maps from an int to the real bytes list
    # this avoids changing the untouched pairs in pair_to_bytes_lst
    id_to_tokens = {}
    word_id_counts = {}

    merges = []

    for i, (key, val) in enumerate(match_freq_table.items()):
        bytes_lst = list(vocab_table[bytes([x])] for c in key for x in c.encode('utf-8'))
        id_to_tokens[i] = tuple(bytes_lst)
        # add a layer of abstraction. 
        # key in bytes_counts now is the word_id instead of the real bytes_lst
        # will make the maintain easier
        word_id_counts[i] = val

    word_ids_merged = word_id_counts.keys()
    # record the pair to bytes_lst and the reverse mapping
    # so that we only need to recompute the byte pairs from bytes_lst that have most_freq_pair
    pair_to_word_ids = defaultdict(set)   
    pair_counts = defaultdict(int)
    word_pair_counts = defaultdict(lambda: defaultdict(int))
    most_freq_pair = None
    
    # init pair_counts, pair_to_word_ids, and word_pair_counts
    for word_id, tokens in id_to_tokens.items():
        frequency = word_id_counts[word_id]
        for pair in zip(tokens, tokens[1:]):
            pair_counts[pair] += frequency
            pair_to_word_ids[pair].add(word_id)
            word_pair_counts[word_id][pair] += 1

    candidates = None
    token_rank = {
        token_id: reverse_lex_key(token)
        for token_id, token in reverse_vocab_table.items()
    }
    for iteration in tqdm(range(iterations), desc='Merge', disable=not show_prograss):
        if candidates is None:
            # use -count to make a min heap a max heap
            candidates = [
                entry(pair, count, token_rank) 
                for pair, count in pair_counts.items() if count > 0
            ]
            heapq.heapify(candidates)
        
        while candidates:
            neg_count, _, _, left_id, right_id = heapq.heappop(candidates)
            pair = (left_id, right_id)
            if pair_counts.get(pair, 0) == -neg_count:
                most_freq_pair = pair
                break
        else:
            break

        # after finding the most frequent byte pairs,
        # update vocab_table and bytes_count
        converted = b''.join([reverse_vocab_table[x] for x in most_freq_pair])
        new_id = len(vocab_table)
        vocab_table[converted] = new_id
        reverse_vocab_table[new_id] = converted
        token_rank[new_id] = reverse_lex_key(converted)

        merges.append((
            reverse_vocab_table[most_freq_pair[0]],
            reverse_vocab_table[most_freq_pair[1]]
        ))
        word_ids_merged = pair_to_word_ids[most_freq_pair]
        word_id_counts_merged = {
            word_id_merged: word_id_counts[word_id_merged] for word_id_merged in word_ids_merged
        }

        affected = dict(word_id_counts_merged)
        pairs_affected = set()
        for word_id, frequency in affected.items():
            old_tokens = id_to_tokens[word_id]
            new_tokens, removed_pairs, added_pairs = merge_exact_pair(old_tokens, most_freq_pair, len(vocab_table)-1)
            
            id_to_tokens[word_id] = new_tokens
            pairs_affected.update(removed_pairs)
            
            for pair, count in removed_pairs.items():
                pair_counts[pair] -= frequency * count

                new_candidate = entry(pair, pair_counts[pair], token_rank)
                heapq.heappush(candidates, new_candidate)
                word_pair_counts[word_id][pair] -= count
                if word_pair_counts[word_id][pair] == 0:
                    pair_to_word_ids[pair].discard(word_id)
            for pair, count in added_pairs.items():
                pair_counts[pair] += frequency * count

                new_candidate = entry(pair, pair_counts[pair], token_rank)
                heapq.heappush(candidates, new_candidate)

                word_pair_counts[word_id][pair] += count
                pair_to_word_ids[pair].add(word_id)

        # remove those that have 0 freq (e.g. most_freq_pair)
        pair_to_word_ids.pop(most_freq_pair)
        pair_counts.pop(most_freq_pair)
        for pair in pairs_affected:
            if pair_counts[pair] == 0:
                pair_counts.pop(pair)

    return reverse_vocab_table, merges

def merge_exact_pair(
    old_tokens: tuple[bytes], 
    most_freq_pair: tuple[bytes], 
    new_token_id: int, 
):
    i = 0
    result = []
    while i < len(old_tokens):
        if i + 1 < len(old_tokens) and old_tokens[i: i+2] == most_freq_pair:
            result.append(new_token_id)
            first_occur_idx = i
            i += 2
        else:
            result.append(old_tokens[i])
            i += 1
    new_tokens = tuple(result)
    
    # core
    old_pairs = Counter(zip(old_tokens, old_tokens[1:]))
    new_pairs = Counter(zip(new_tokens, new_tokens[1:]))
    removed_pairs = old_pairs - new_pairs
    added_pairs = new_pairs - old_pairs
    return new_tokens, removed_pairs, added_pairs

def convert_and_print(items: list | dict, reverse_vocab_table):
    if isinstance(items, dict):
        return {tuple([reverse_vocab_table[x] for x in key]):val for key, val in items.items()}
    return [[reverse_vocab_table[x] for x in key] for key in items]

def test(text:str):
    vocab_table = {}
    init_vocab(vocab_table)
    match_freq_table = pretokenize(text)
    merge(vocab_table, match_freq_table, 6)
    breakpoint()

def main(vocab_table, file_path):
    pattern = "|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
    with open(file_path, 'rb') as f:
        num_processes = 1
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        match_freq_table = defaultdict(int)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode('utf-8', errors='ignore')
            text_splits = re.split(pattern, chunk) if special_tokens else [chunk]
            
            for text_split in tqdm(text_splits, desc='Pre-tokenization'):
                res = pretokenize(text_split)
                for key, val in res.items():
                    match_freq_table[key] += val
        merge(vocab_table, match_freq_table, 10_000)

def pretokenize_mp(job):
    boundary, file_path, pattern, special_tokens = job
    start, end = boundary
    match_freq_table = defaultdict(int)
    
    start_time = time.perf_counter()
    with open(file_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode('utf-8', errors='ignore')

    text_splits = re.split(pattern, chunk) if special_tokens else [chunk]
    for text_split in text_splits:
        for match in re.finditer(PAT, text_split):
            content = match.group()
            match_freq_table[content] += 1
    elapsed = time.perf_counter() - start_time
    return match_freq_table, elapsed

def main_mp(vocab_table, file_path, special_tokens, vocab_size, show_prograss=False):
    pattern = "|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
    with open(file_path, 'rb') as f:
        num_processes = 64
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        boundaries = list(zip(boundaries[:-1], boundaries[1:]))

    match_freq_table = defaultdict(int)
    ctx = mp.get_context("spawn")
    jobs = [(boundary, file_path, pattern, special_tokens) for boundary in boundaries]
    with ctx.Pool(processes=num_processes) as pool:
        with tqdm(total=len(jobs), desc='Pre-tokenize', unit='chunk', disable=not show_prograss) as progress:
            for local_counts, elapsed in pool.imap_unordered(pretokenize_mp, jobs):
                for token, count in local_counts.items():
                    match_freq_table[token] += count

                progress.update(1)
                progress.set_postfix(last=f'{elapsed:.2f}')

    vocab, merges = merge(
        vocab_table, 
        match_freq_table, 
        iterations = vocab_size - len(vocab_table),
        show_prograss=show_prograss
    )

    return vocab, merges

def train_BPETokenizer(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    show_prograss=False,
    show_memory_peak=False
):
    if show_memory_peak:
        tracemalloc.start()
    vocab_table = {}
    init_vocab(vocab_table)
    vocab, merges = main_mp(
        vocab_table, input_path, special_tokens, vocab_size, show_prograss
    )
    if show_memory_peak:
        cur, peak = tracemalloc.get_traced_memory()
        print(f"Peak memory usage: {peak/1024**3} GB")
        tracemalloc.end()
    return vocab, merges

if __name__ == '__main__':
    # test(TEST_TEXT)

    tracemalloc.start()
    vocab_table = {}
    init_vocab(vocab_table)

    special_tokens = ['<|endoftext|>']
    tiny_story_train_path = '/global/cfs/cdirs/m4410/mzheng/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt'
    tiny_story_valid_path = '/global/cfs/cdirs/m4410/mzheng/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt'
    owt_train_path = '/global/cfs/cdirs/m4410/mzheng/cs336/assignment1-basics/data/owt_train.txt'
    # main(vocab_table, tiny_story_valid_path, special_tokens)
    # main_mp(vocab_table, tiny_story_valid_path, special_tokens)
    # main_mp(vocab_table, tiny_story_train_path, special_tokens)
    main_mp(vocab_table, owt_train_path, special_tokens)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
