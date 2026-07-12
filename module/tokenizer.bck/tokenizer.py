import multiprocessing as mp 
import json
from collections.abc import Iterable, Iterator
from collections import deque
import regex as re
from functools import lru_cache
import heapq
from typing import List

@lru_cache()
def bytes_to_unicode(): # copied from oai's gpt2 repo
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

class Tokenizer:
    def __init__(self,
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens:list[str] | None = None):
        '''
        PAT for pre-tokenization: breaks a text into 'words', which then are tokenized into subword tokens
        '''
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.decode_vocab = vocab # tokenID -> bytes, handy for decode, but not for encode
        self.encode_vocab = {val: key for key, val in vocab.items()}
        self.merges = merges
        self.merge_ranks = {merge: i for i, merge in enumerate(merges)}
        self.special_tokens = special_tokens
        self.word_re = re.compile(self.PAT)
        if special_tokens:
            ordered_specials = sorted(special_tokens, key=len, reverse=True)
            self.special_re = re.compile("|".join(map(re.escape, ordered_specials)))
        else:
            self.special_re = None


    @classmethod
    def from_files(cls,
                   vocab_filepath: str,
                   merge_filepath: str,
                   special_tokens: list[str] | None = None):
        byte_decoder = {v: k for k, v in bytes_to_unicode().items()}

        def decode_gpt2_token(s: str) -> bytes:
            return bytes([byte_decoder[ch] for ch in s])
        
        vocab, merges = {}, []
        
        with open(vocab_filepath, 'r') as f:
            vocab_json = json.load(f) # need to convert the key from str to bytes
        for key, val in vocab_json.items():
            if special_tokens is not None and key in special_tokens:
                continue
            vocab[val] = decode_gpt2_token(key)

        with open(merge_filepath, 'r') as f:
            merges_txt = f.read()
        for merge in merges_txt.split('\n'):
            ms = merge.split(' ')
            if len(ms) < 2:
                continue
            merges.append((decode_gpt2_token(ms[0]), decode_gpt2_token(ms[1])))
        return Tokenizer(vocab, merges, special_tokens)
    
    def get_workload_spans(self, target_bytes=1_000):
        n = 0
        idx = 0
        start = 0
        for word in self.words:
            byte_seq = word.encode('utf-8')
            n += len(byte_seq)
            idx += 1
            if n >= target_bytes:
                yield (start, idx)
                start = idx
        yield (start, idx)


    def _iter_non_special_spans(self, text: str):
        if not self.special_re:
            yield 0, len(text), None
            return
        last = 0
        for m in self.special_re.finditer(text):
            if last < m.start():
                yield last, m.start(), None
            yield m.start(), m.end(), m.group()
            last = m.end()
        if last < len(text):
            yield last, len(text), None

    def _encode_word(self, word: str) -> List[int]:
        if not word:
            return []

        tok_bytes: List[bytes | None] = [bytes([b]) for b in word.encode("utf-8")]
        n = len(tok_bytes)
        if n == 1:
            return [self.encode_vocab[tok_bytes[0]]]  # type: ignore[index]

        prev = [-1] + list(range(n - 1))
        nxt = list(range(1, n)) + [-1]
        head = 0

        heap = []
        i = head
        while i != -1:
            j = nxt[i]
            if j != -1:
                pair = (tok_bytes[i], tok_bytes[j])  # type: ignore[arg-type]
                rank = self.merge_ranks.get(pair)
                if rank is not None:
                    heapq.heappush(heap, (rank, i, j))
            i = nxt[i]
        
        while heap:
            rank, i, j = heapq.heappop(heap)
            if i == -1 or j == -1:
                continue
            if nxt[i] != j:
                continue  # not adjacent anymore
            if tok_bytes[i] is None or tok_bytes[j] is None:
                continue

            merged = tok_bytes[i] + tok_bytes[j]
            breakpoint()

            k = len(tok_bytes)
            tok_bytes.append(merged)
            prev.append(-1)
            nxt.append(-1)

            left, right = prev[i], nxt[j]
            if left != -1:
                nxt[left] = k
                prev[k] = left
            else:
                head = k
            if right != -1:
                prev[right] = k
                nxt[k] = right

            tok_bytes[i] = None
            tok_bytes[j] = None
            prev[i] = nxt[i] = prev[j] = nxt[j] = -1

            if left != -1:
                pair = (tok_bytes[left], tok_bytes[k])  # type: ignore[arg-type]
                rank = self.merge_ranks.get(pair)
                if rank is not None and pair[0] is not None and pair[1] is not None:
                    heapq.heappush(heap, (rank, left, k))
            if right != -1:
                pair = (tok_bytes[k], tok_bytes[right])  # type: ignore[arg-type]
                rank = self.merge_ranks.get(pair)
                if rank is not None and pair[0] is not None and pair[1] is not None:
                    heapq.heappush(heap, (rank, k, right))

        ids: List[int] = []
        i = head
        while i != -1:
            token = tok_bytes[i]
            if token is not None:
                ids.append(self.encode_vocab[token])
            i = nxt[i]
        return ids

    def encode(self, text:str) -> list[int]:
        '''
        Encode an input text into a sequence of token IDs
        '''
        ids: List[int] = []
        for start, end, special in self._iter_non_special_spans(text):
            if special is not None:
                ids.append(self.encode_vocab[special.encode("utf-8")])
                continue
            chunk = text[start:end]
            for m in self.word_re.finditer(chunk):
                ids.extend(self._encode_word(m.group()))
        return ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for _id in self.encode(text):
                yield _id
    
    def decode(self, ids: list[int]) -> str:
        '''
        Decode a sequence of token IDs into text
        '''
        results = []
        for id in ids:
            results.append(self.decode_vocab[id])
        return b"".join(results).decode('utf-8', errors='replace')

if __name__ == '__main__':
    merge_filepath = '/global/cfs/cdirs/m4410/mzheng/cs336/assignment1-basics/tests/fixtures/train-bpe-reference-merges.txt'
    vocab_filepath = '/global/cfs/cdirs/m4410/mzheng/cs336/assignment1-basics/tests/fixtures/train-bpe-reference-vocab.json'

    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_filepath,
        merge_filepath=merge_filepath,
        special_tokens=['<|endoftext|>']
    )

    # test_filepath = '/global/cfs/cdirs/m4410/mzheng/cs336/assignment1-basics/tests/fixtures/corpus.en'
    # test_str = None
    # with open (test_filepath, 'r') as f:
    #     test_str = f.read()
    test_str = "Hello, how are you?"
    encoded = tokenizer.encode(test_str)
    decoded = tokenizer.decode(encoded)
    assert decoded == test_str
