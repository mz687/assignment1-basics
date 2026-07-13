from .BPETokenizer import BPETokenizer
import regex as re

owt_vocab_path = '/global/cfs/cdirs/m4410/mzheng/cs336/assignment1-basics/data/owt_train_vocab.json'
owt_merges_path = '/global/cfs/cdirs/m4410/mzheng/cs336/assignment1-basics/data/owt_train_merges.txt'
owt_valid_path = '/global/cfs/cdirs/m4410/mzheng/cs336/assignment1-basics/data/owt_valid.txt'

tiny_story_vocab_path = '/global/cfs/cdirs/m4410/mzheng/cs336/assignment1-basics/data/TinyStoriesV2-GPT4_vocab.json'
tiny_story_merges_path = '/global/cfs/cdirs/m4410/mzheng/cs336/assignment1-basics/data/TinyStoriesV2-GPT4_merges.txt'
tiny_story_valid_path = '/global/cfs/cdirs/m4410/mzheng/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt'

special_tokens = ['<|endoftext|>']
pattern = "|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True))

def sample_from_corpus(file_path: str, n_samples: int=10) -> list[str]:
    import os, random
    random.seed(42)

    assert os.path.exists(file_path), print(f"file_path ({file_path}) cannot be found!")

    with open(file_path, 'r', encoding='utf-8') as f:
        text = "".join(l for l in f.readlines())
    text_splits = re.split(f"({pattern})", text)
    idx = [random.randint(0, len(text_splits)) for _ in range(n_samples)]
    return [text_splits[i] for i in idx]

def main():
    owt_tokenizer = BPETokenizer.from_files(
        vocab_file_path=owt_vocab_path,
        merges_file_path=owt_merges_path,
        special_tokens=special_tokens
    )

    tiny_story_tokenizer = BPETokenizer.from_files(
        vocab_file_path=tiny_story_vocab_path,
        merges_file_path=tiny_story_merges_path,
        special_tokens=special_tokens
    )

    owt_valid_samples = sample_from_corpus(
        file_path=owt_valid_path,
        n_samples=10
    )
    owt_total_bytes = 0
    owt_total_tokens = 0
    for sample in owt_valid_samples:
        owt_total_bytes += len(sample.encode('utf-8'))
        owt_total_tokens += len(owt_tokenizer.encode(sample))

    tiny_story_valid_samples = sample_from_corpus(
        file_path=tiny_story_valid_path,
        n_samples=10
    )
    tiny_story_total_bytes = 0
    tiny_story_total_tokens = 0
    for sample in tiny_story_valid_samples:
        tiny_story_total_bytes += len(sample.encode('utf-8'))
        tiny_story_total_tokens += len(tiny_story_tokenizer.encode(sample))

    print(f'owt_total_bytes: {owt_total_bytes}, owt_total_tokens: {owt_total_tokens}, compression ratio (bytes/token): {owt_total_bytes/owt_total_tokens}')
    print(f'tiny_story_total_bytes: {tiny_story_total_bytes}, tiny_story_total_tokens: {tiny_story_total_tokens}, compression ratio (bytes/token): {tiny_story_total_bytes/tiny_story_total_tokens}')

    owt_total_bytes = 0
    owt_total_tokens = 0
    for sample in owt_valid_samples:
        owt_total_bytes += len(sample.encode('utf-8'))
        owt_total_tokens += len(tiny_story_tokenizer.encode(sample))
    print('\n')
    print(f'[tiny_story_tokenizer tokenizes owt samples] owt_total_bytes: {owt_total_bytes}, owt_total_tokens: {owt_total_tokens}, compression ratio: {owt_total_bytes/owt_total_tokens}')


if __name__ == '__main__':
    main()


