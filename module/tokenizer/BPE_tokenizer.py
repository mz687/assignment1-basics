r'''
To convert any unicode to char or the other way around:
ord('牛‘) -> get unicode 29275
char(29275) -> get 牛

Understanding Unicode
(a) chr(0) -> '\x00'
(b) chr(0).__repr__() -> "'\\x00'"
(c) "this is a test" + chr(0) + "string" -> 'this is a test\x00string'
    print("this is a test" + chr(0) + "string") -> "this is a teststring"

Unicode Encoding
(a) UTF-8 is enough for most cases, and compared to UTF-16 and UTF-32, the encoded number of bytes is shorter.
(b) It cannot correctly decode more than 1 byte sequence
(c) \xff\xff

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
Use re.finditer instead of re.findall to use iterator rather than storing all the results
'''

import regex as reg
import heapq
from collections import Counter, defaultdict

class BPETokenizer:
    def __init__(self, input_path: str, vocab_size: int, special_tokens: list[str]):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.word_re = reg.compile(self.PAT)

        # split points for special tokens (do not train across them)
        self.chunk_re = reg.compile("|".join(map(reg.escape, special_tokens))) if special_tokens else None

        # token bytes table (id -> bytes). Keep special tokens out of training.
        self.tok_bytes = []
        for st in special_tokens:
            self.tok_bytes.append(st.encode("utf-8"))  # store as bytes for uniformity

        self.byte_offset = len(self.tok_bytes)
        for b in range(256):
            self.tok_bytes.append(bytes([b]))

        self.merges = []

    @staticmethod
    def _rev_lex_key(token_bytes: bytes) -> tuple[int, ...]:
        """
        Build a key where Python's default ascending tuple comparison corresponds to
        descending lexicographic order of the original byte string.
        """
        return tuple(-b for b in token_bytes) + (1,)

    def _build_spans(self, text: str):
        """Return [(start,end), ...] spans excluding special tokens."""
        if not self.chunk_re:
            return [(0, len(text))]
        spans = []
        last = 0
        for m in self.chunk_re.finditer(text):
            spans.append((last, m.start()))
            last = m.end()
        spans.append((last, len(text)))
        return spans

    def _count_word_types(self, text: str, spans):
        """
        Count unique pre-tokenized pieces (bytes) across corpus.
        Key optimization: operate on word *types* not all occurrences.
        """
        wc = Counter()
        finditer = self.word_re.finditer
        for s, e in spans:
            for m in finditer(text, s, e):
                wc[m.group().encode("utf-8")] += 1
        return wc

    def train(self):
        with open(self.input_path, "r", encoding="utf-8") as f:
            text = f.read()

        spans = self._build_spans(text)
        word_counts = self._count_word_types(text, spans)

        # ----- Build per-word linked lists (arrays) -----
        # Node arrays:
        tok = []      # token id at node
        nxt = []      # next node index, -1 if end
        prv = []      # prev node index, -1 if head
        wid = []      # word id for node (to get word frequency)

        word_freq = []   # word_freq[word_id]
        heads = []       # head node index per word_id

        def new_node(tid, prev_i, word_id):
            i = len(tok)
            tok.append(tid)
            prv.append(prev_i)
            nxt.append(-1)
            wid.append(word_id)
            if prev_i != -1:
                nxt[prev_i] = i
            return i

        # Build nodes only for unique word types
        for w_id, (w_bytes, freq) in enumerate(word_counts.items()):
            word_freq.append(freq)
            prev_i = -1
            head = -1
            for b in w_bytes:
                tid = self.byte_offset + b  # map byte to token id
                i = new_node(tid, prev_i, w_id)
                if head == -1:
                    head = i
                prev_i = i
            heads.append(head)

        # ----- Initialize pair counts and occurrence lists -----
        pair_count = Counter()
        pair_occ = defaultdict(list)  # (a,b) -> [node_i, node_i, ...] where node_i is first token in pair

        for w_id, head in enumerate(heads):
            if head == -1:
                continue
            weight = word_freq[w_id]
            i = head
            j = nxt[i]
            while j != -1:
                p = (tok[i], tok[j])
                pair_count[p] += weight
                pair_occ[p].append(i)
                i = j
                j = nxt[j]

        tok_rank = [self._rev_lex_key(tb) for tb in self.tok_bytes]

        # Max-heap by count, then by lexicographically greatest raw-byte pair.
        # We use reverse lex keys so heapq's min-order picks the desired maximum.
        heap = [(-c, tok_rank[a], tok_rank[b], a, b) for (a, b), c in pair_count.items() if c > 0]
        heapq.heapify(heap)

        # ----- Incremental merges -----
        while len(self.tok_bytes) < self.vocab_size and heap:
            negc, _ra, _rb, a, b = heapq.heappop(heap)
            ccur = -negc
            if pair_count.get((a, b), 0) != ccur or ccur <= 0:
                continue  # stale heap entry

            # create merged token
            new_id = len(self.tok_bytes)
            self.tok_bytes.append(self.tok_bytes[a] + self.tok_bytes[b])
            tok_rank.append(self._rev_lex_key(self.tok_bytes[new_id]))
            self.merges.append((self.tok_bytes[a], self.tok_bytes[b]))  # store bytes-level merge if you want

            occ_list = pair_occ.pop((a, b), [])
            # Merge application must treat all existing (a,b) occurrences as consumed.
            # This avoids over-counting from stale/overlapping entries in occ_list.
            pair_count[(a, b)] = 0
            occ_list.sort()

            for i in occ_list:
                j = nxt[i]
                if j < 0 or tok[i] != a or tok[j] != b or prv[j] != i:
                    continue  # stale occurrence
                w = word_freq[wid[i]]
                li = prv[i]
                r  = nxt[j]

                # remove old neighboring pairs
                if li != -1:
                    oldL = (tok[li], a)
                    if pair_count.get(oldL, 0):
                        pair_count[oldL] -= w
                        heapq.heappush(
                            heap,
                            (-pair_count[oldL], tok_rank[oldL[0]], tok_rank[oldL[1]], oldL[0], oldL[1]),
                        )
                if r != -1:
                    oldR = (b, tok[r])
                    if pair_count.get(oldR, 0):
                        pair_count[oldR] -= w
                        heapq.heappush(
                            heap,
                            (-pair_count[oldR], tok_rank[oldR[0]], tok_rank[oldR[1]], oldR[0], oldR[1]),
                        )

                # perform merge: replace tok[i] with new_id and delete node j from chain
                tok[i] = new_id
                nxt[i] = r
                if r != -1:
                    prv[r] = i
                # Mark removed node as inactive so stale pointers never look valid.
                prv[j] = -2
                nxt[j] = -2

                # add new neighboring pairs
                if li != -1:
                    newL = (tok[li], new_id)
                    pair_count[newL] += w
                    pair_occ[newL].append(li)
                    heapq.heappush(
                        heap,
                        (-pair_count[newL], tok_rank[newL[0]], tok_rank[newL[1]], newL[0], newL[1]),
                    )
                if r != -1:
                    newR = (new_id, tok[r])
                    pair_count[newR] += w
                    pair_occ[newR].append(i)
                    heapq.heappush(
                        heap,
                        (-pair_count[newR], tok_rank[newR[0]], tok_rank[newR[1]], newR[0], newR[1]),
                    )

            # optional cleanup
            if pair_count[(a, b)] <= 0:
                pair_count.pop((a, b), None)

        # Build vocab dict like your original API (id -> bytes)
        vocab = {i: b for i, b in enumerate(self.tok_bytes)}
        return vocab, self.merges
