"""Extensible Zero-[UNK] WordPiece + UTF-8 Byte-Fallback Tokenizer and Length-Bucket Packer for MeguvizCAC v0.3-Loop."""

import json
import unicodedata
from pathlib import Path

import numpy as np


MAX_LEN = 192
HEAD_CHARS = 140
TAIL_CHARS = 50
BASE_VOCAB_SIZE = 21128
BYTE_TOKENS = 256
TOTAL_VOCAB_SIZE = BASE_VOCAB_SIZE + BYTE_TOKENS


def prepare_text(text: str) -> str:
    """Normalize and apply head+tail truncation keeping both openers and closers within 192 chars."""
    t = " ".join(unicodedata.normalize("NFKC", text).split())
    if len(t) > MAX_LEN:
        t = t[:HEAD_CHARS] + " " + t[-TAIL_CHARS:]
    return t


def _is_punct(ch: str) -> bool:
    cp = ord(ch)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(ch).startswith("P")


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0xF900 <= cp <= 0xFAFF
    )


def _is_control(ch: str) -> bool:
    if ch in ("\t", "\n", "\r"):
        return False
    cat = unicodedata.category(ch)
    return cat.startswith("C")


class ByteFallbackWordPieceTokenizer:
    """WordPiece tokenizer 100% ID-compatible with BAAI/bge-small-zh-v1.5 + 256 UTF-8 byte fallback slots."""

    def __init__(self, vocab_path: Path, lowercase: bool = False, byte_offset: int = BASE_VOCAB_SIZE):
        self.vocab_path = Path(vocab_path)
        self.lowercase = lowercase
        self.byte_offset = int(byte_offset)
        tj_path = self.vocab_path.parent / "tokenizer.json"
        if tj_path.is_file():
            tj = json.loads(tj_path.read_text(encoding="utf-8"))
            self.vocab: dict[str, int] = dict(tj["model"]["vocab"])
            norm = tj.get("normalizer") or {}
            if "lowercase" in norm:
                self.lowercase = bool(norm["lowercase"])
        else:
            self.vocab = {}
            with self.vocab_path.open("r", encoding="utf-8", newline="\n") as f:
                for idx, line in enumerate(f):
                    tok = line.rstrip("\r\n")
                    if tok == "" and idx >= BASE_VOCAB_SIZE:
                        continue
                    self.vocab[tok] = idx

        self.pad_id = int(self.vocab.get("[PAD]", 0))
        self.unk_id = int(self.vocab["[UNK]"])
        self.cls_id = int(self.vocab["[CLS]"])
        self.sep_id = int(self.vocab["[SEP]"])
        self.vocab_size = self.byte_offset + BYTE_TOKENS

    def _basic_tokenize(self, text: str) -> list[str]:
        if self.lowercase:
            text = text.lower()
        spaced: list[str] = []
        for ch in text:
            if _is_control(ch):
                continue
            if _is_cjk(ch) or _is_punct(ch):
                spaced.extend((" ", ch, " "))
            elif ch.isspace():
                spaced.append(" ")
            else:
                spaced.append(ch)
        return "".join(spaced).split()

    def encode_ids(self, text: str, max_len: int = MAX_LEN, return_stats: bool = False):
        """Encode prepared text into [CLS] ... [SEP] token IDs without ever emitting [UNK]."""
        ids = [self.cls_id]
        byte_fallbacks = 0
        budget = max_len - 1

        for tok in self._basic_tokenize(prepare_text(text)):
            if len(ids) >= budget:
                break
            if tok in self.vocab:
                ids.append(self.vocab[tok])
                continue

            start = 0
            subword_cont = False
            while start < len(tok) and len(ids) < budget:
                end = len(tok)
                found_id = None
                while start < end:
                    sub = ("##" + tok[start:end]) if subword_cont else tok[start:end]
                    if sub in self.vocab:
                        found_id = self.vocab[sub]
                        break
                    end -= 1
                if found_id is not None:
                    ids.append(found_id)
                    start = end
                    subword_cont = True
                else:
                    b_ids = [self.byte_offset + b for b in tok[start].encode("utf-8")]
                    if len(ids) + len(b_ids) > budget:
                        ids = ids[:budget]
                        start = len(tok)
                        break
                    ids.extend(b_ids)
                    byte_fallbacks += len(b_ids)
                    start += 1
                    subword_cont = False

        ids = ids[:budget] + [self.sep_id]
        if return_stats:
            return ids, {
                "seq_len": len(ids),
                "byte_fallback_tokens": byte_fallbacks,
                "unk_tokens": 0,
            }
        return ids

    def encode(self, text: str, max_len: int = MAX_LEN) -> tuple[np.ndarray, np.ndarray]:
        """Encode a single text string into (1, seq_len) int64 input_ids and attention_mask."""
        ids = self.encode_ids(text, max_len=max_len)
        mask = [1] * len(ids)
        return np.array([ids], dtype=np.int64), np.array([mask], dtype=np.int64)

    def encode_batch(
        self,
        texts: list[str],
        max_len: int = MAX_LEN,
        pad_to_multiple: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Encode a list of texts into padded int64 numpy arrays plus raw unpadded lengths."""
        if not texts:
            empty = np.zeros((0, 0), dtype=np.int64)
            return empty, empty, []
        encoded = [self.encode_ids(t, max_len=max_len) for t in texts]
        lengths = [len(seq) for seq in encoded]
        max_seq = max(lengths)
        if pad_to_multiple > 1:
            rem = max_seq % pad_to_multiple
            if rem:
                max_seq = min(max_len, max_seq + (pad_to_multiple - rem))
        ids_arr = np.full((len(encoded), max_seq), self.pad_id, dtype=np.int64)
        mask_arr = np.zeros((len(encoded), max_seq), dtype=np.int64)
        for i, seq in enumerate(encoded):
            n = len(seq)
            ids_arr[i, :n] = seq
            mask_arr[i, :n] = 1
        return ids_arr, mask_arr, lengths


def pack_length_buckets(
    lengths: list[int],
    max_batch: int = 32,
    boundaries: tuple[int, ...] = (32, 64, 128, 192),
) -> list[list[int]]:
    """Group sequence indices by tightest length bucket, sort within bucket, and slice into <= max_batch."""
    if not lengths:
        return []
    if max_batch <= 0:
        raise ValueError("max_batch must be positive")
    sorted_bounds = tuple(sorted(boundaries))
    buckets: dict[int, list[int]] = {b: [] for b in sorted_bounds}
    overflow_key = sorted_bounds[-1] + 1
    buckets[overflow_key] = []

    for idx, length in enumerate(lengths):
        placed = False
        for b in sorted_bounds:
            if length <= b:
                buckets[b].append(idx)
                placed = True
                break
        if not placed:
            buckets[overflow_key].append(idx)

    micro_batches: list[list[int]] = []
    for key in (*sorted_bounds, overflow_key):
        idxs = buckets[key]
        if not idxs:
            continue
        idxs.sort(key=lambda i: lengths[i])
        for start in range(0, len(idxs), max_batch):
            micro_batches.append(idxs[start : start + max_batch])
    return micro_batches
