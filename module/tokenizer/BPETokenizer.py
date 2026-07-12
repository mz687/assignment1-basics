import os
import regex as re
import multiprocessing as mp
from .common import gpt2_bytes_to_unicode
import json
from pathlib import Path
from collections.abc import Iterable, Iterator
import heapq

class BPETokenizer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(
        self, 
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.reverse_vocab = {token: ID for ID, token in vocab.items()}
        self.merges = merges
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)} 
        self.special_tokens = special_tokens if special_tokens else []
        self.pattern = "|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True))
    
    @classmethod
    def from_files(
        cls,
        vocab_file_path: str,
        merges_file_path: str,
        special_tokens: list[str] | None  = None
    ):
        '''
        read in the vocab (dict[int, bytes]) and the merges (list[tuple(bytes, bytes)]).
        remember to decode the bytes in vocab, which were encode by gpt2_bytes_to_unicode so invalid bytes can be stored.
        '''
        assert os.path.exists(Path(vocab_file_path)), f"vocab_file_path ({vocab_file_path}) not exist"
        assert os.path.exists(Path(merges_file_path)), f"merges_file_path ({merges_file_path}) not exist"

        special_tokens_set = set(special_tokens)
        byte_encoder = gpt2_bytes_to_unicode()
        byte_decoder = {val: key for key, val in byte_encoder.items()}
        def decode_token(text: str) -> bytes:
            if text in special_tokens_set:
                return text.encode('utf-8')
            return bytes([byte_decoder[x] for x in text])

        with Path(vocab_file_path).open('r', encoding='utf-8') as f:
            vocab = json.load(f)
        vocab = {
            ID: decode_token(text)
            for text, ID in vocab.items()
        }
        
        merges = []
        with Path(merges_file_path).open('r', encoding='utf-8') as f:
            for line in f:
                str1, str2 = line.replace('\n','').split(' ')
                merges.append((
                    decode_token(str1),
                    decode_token(str2)
                ))

        return cls(
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens
        )
    
    def _encode_word(self, word: str) -> list[int]:
        '''
        break word into utf-8 bytes and then follow the sequence in merges (smallest self.merge_ranks) 
        to gradually merge the pairs until they cannot be merged.
        finally, return a list of int (ids in self.vocab)
        '''
        tokens = [
            bytes([x]) 
            for x in word.encode('utf-8')
        ]
        while True:
            if len(tokens) < 2:
                break
            best_pair = min(
                (
                    (self.merge_ranks[pair], pair) 
                    for pair in zip(tokens, tokens[1:]) 
                    if pair in self.merge_ranks
                ),
                default=None
            )
            if best_pair is None:
                break
            best_pair = best_pair[1]
            merged = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and \
                    (tokens[i], tokens[i+1]) == best_pair:
                    merged.append((tokens[i] + tokens[i+1]))
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            tokens = merged 
        return [self.reverse_vocab[token] for token in tokens]

    def encode(self, text: str) -> list[int]:
        '''
        encode a str to a list of token ids.
        '''
        ret = []
        text_splits = re.split(f"({self.pattern})", text) if self.special_tokens else [text]
        for text_split in text_splits:
            if text_split in self.special_tokens:
                ret.append(self.reverse_vocab[text_split.encode('utf-8')])
                continue
            for match in re.finditer(self.PAT, text_split):
                content = match.group()
                ret.extend(self._encode_word(content))
        return ret
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
    
    def decode(self, ids: list[int]) -> str:
        '''
        decode a sequence of token ids to str.
        '''
        return b"".join(self.vocab[ID] for ID in ids).decode('utf-8', errors='replace')

if __name__ == '__main__':
    tokenizer = BPETokenizer.from_files(
        merges_file_path='/global/cfs/cdirs/m4410/mzheng/cs336/assignment1-basics/data/owt_train_merges.txt',
        vocab_file_path='/global/cfs/cdirs/m4410/mzheng/cs336/assignment1-basics/data/owt_train_vocab.json',
        special_tokens=["<|endoftext|>"]
    )

    text = "This is an example<|endoftext|>"
    # text = "🙃"
    print(tokenizer.encode_iterable(text))
    print(tokenizer.decode(tokenizer.encode(text)))