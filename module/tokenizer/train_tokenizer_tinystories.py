from .tokenizer import train_BPETokenizer
import json
from pathlib import Path
from .common import gpt2_bytes_to_unicode

def save(vocab, merges, vocab_path, merges_path):
    byte_encoder = gpt2_bytes_to_unicode()

    def encode_token(token: bytes) -> str:
        return "".join(byte_encoder[byte] for byte in token)
    
    vocab_json = {
        encode_token(token): ID for ID, token in vocab.items()
    }
    with Path(vocab_path).open('w', encoding='utf-8') as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)
    print(f"vocab has been saved to {vocab_path}.")

    with Path(merges_path).open('w', encoding='utf-8', newline='\n') as f:
        for left, right in merges:
            f.write(
                f"{encode_token(left)} {encode_token(right)}\n"
            )
    print(f"merges have been saved to {merges_path}.")

def main():
    tiny_story_train_path = '/global/cfs/cdirs/m4410/mzheng/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt'
    vocab_size = 10_000 # train on tiny_story for 10k steps
    special_tokens = ['<|endoftext|>']

    vocab, merges = train_BPETokenizer(
        input_path=tiny_story_train_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_prograss=True,
    )

    vocab_path = '/global/cfs/cdirs/m4410/mzheng/cs336/assignment1-basics/data/TinyStoriesV2-GPT4_vocab.json'
    merges_path = '/global/cfs/cdirs/m4410/mzheng/cs336/assignment1-basics/data/TinyStoriesV2-GPT4_merges.txt'
    save(
        vocab=vocab,
        merges=merges,
        vocab_path=vocab_path,
        merges_path=merges_path
    )

if __name__ == '__main__':
    main()

