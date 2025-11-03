import os
import sys
import json
import heapq
import regex as re
from typing import Iterable, Iterator
from collections import OrderedDict
from memory_profiler import profile

from .utils import gpt2_bytes_to_unicode, PAT


class BPETokenizer:
    def __init__(
            self,
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] | None = None
    ):
        self.gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        self.gpt2_byte_encoder = {k: v for k, v in gpt2_bytes_to_unicode().items()}
        self.merges = {merge: idx for idx, merge in enumerate(merges)}
        self.special_tokens = special_tokens
        self.vocab_encoder = {v: k for k, v in vocab.items()} # bytes -> id
        if special_tokens is not None:
            for special in special_tokens:
                special_encoded = special.encode()
                if special_encoded not in self.vocab_encoder:
                    self.vocab_encoder[special_encoded] = len(self.vocab_encoder)
        self.vocab_decoder = {v: k for k, v in self.vocab_encoder.items()} # id -> bytes

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ):
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath, 'r', encoding="utf-8") as f:
            vocab = json.load(f)
            vocab = {
                index: bytes([gpt2_byte_decoder[token] for token in vocab_item])
                for vocab_item, index in vocab.items()
            }
        with open(merges_filepath, encoding="utf-8") as f:
            merges = [tuple(line.rstrip().split(" ")) for line in f]
            merges = [
                (
                    bytes([gpt2_byte_decoder[token] for token in merge_token1]),
                    bytes([gpt2_byte_decoder[token] for token in merge_token2]),
                )
                for merge_token1, merge_token2 in merges
            ]
        return BPETokenizer(vocab, merges, special_tokens)

    def encode_pretokened(self, pretoken: list[bytes]) -> list[int]:
        result = []
        for tokens in pretoken:
            heap = []
            for first, second in zip(tokens[:-1], tokens[1:]):
                pair = (first.to_bytes(), second.to_bytes())
                if pair not in self.merges:
                    continue
                heapq.heappush(heap, (self.merges[pair], pair[0], pair[1]))
            pair = None
            encoded = [tok.to_bytes() for tok in tokens]
            while len(heap) != 0:
                idx, first, second = heapq.heappop(heap)
                i = 0
                temp = []
                while i < len(encoded):
                    if i + 1 < len(encoded) and encoded[i] == first and encoded[i + 1] == second:
                        merged = first + second
                        temp.append(merged)
                        if i - 1 >= 0:
                            if (encoded[i-1], merged) in self.merges:
                                heapq.heappush(heap, (self.merges[(encoded[i-1], merged)], encoded[i-1], merged))
                        if i + 2 < len(encoded):
                            if (merged, encoded[i+2]) in self.merges:
                                heapq.heappush(heap, (self.merges[(merged, encoded[i+2])], merged, encoded[i+2]))
                        i += 2
                    else:
                        temp.append(encoded[i])
                        i += 1
                encoded = temp
                temp = None
            result.extend([self.vocab_encoder[tok] for tok in encoded])
        return result

    def _encode(self, text: str) -> list[int]:
        result = []
        last_pos = 0
        if self.special_tokens is not None:
            pat = "|".join([re.escape(spec) for spec in sorted(self.special_tokens, key=len, reverse=True)])
            for match in re.finditer(pat, text, re.FULLCASE | re.BESTMATCH):
                for token in re.finditer(PAT, text, pos=last_pos, endpos=match.start()):
                    result.extend(self.encode_pretokened([token.group().encode()]))
                spec = match.group()
                result.append(self.vocab_encoder[spec.encode()])
                last_pos = match.end()
            for token in re.finditer(PAT, text, pos=last_pos):
                result.extend(self.encode_pretokened([token.group().encode()]))

        else:
            for token in re.finditer(PAT, text):
                result.extend(self.encode_pretokened([token.group().encode()]))
        return result

    # @profile
    def encode(self, text: str) -> list[int]:
        return self._encode(text)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for part in iterable:
            result = self._encode(part)
            for res in result:
                yield res

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab_decoder[id] for id in ids]).decode(encoding="utf-8", errors="replace")

if __name__ == "__main__":
    FIXTURES_PATH = "/home/tyhcyq/cyq/assignment1-basics/tests/fixtures/"
    VOCAB_PATH = FIXTURES_PATH + "gpt2_vocab.json"
    MERGES_PATH = FIXTURES_PATH + "gpt2_merges.txt"
    print(VOCAB_PATH, MERGES_PATH)
    tokenizer = BPETokenizer.from_files(vocab_filepath=VOCAB_PATH, merges_filepath=MERGES_PATH, special_tokens=["<|endoftext|>"])
    test_string = "Hello, how are you?"
    ids = tokenizer.encode(test_string)
    tokenized_string = [tokenizer.decode([x]) for x in ids]
    print(ids)
    print(tokenized_string)
    # Ensure the double <|endoftext|><|endoftext|> is preserved as a single token
    assert tokenized_string.count("<|endoftext|>") == 1
    assert tokenized_string.count("<|endoftext|><|endoftext|>") == 1
    # Test roundtrip
    assert tokenizer.decode(ids) == test_string
    pass