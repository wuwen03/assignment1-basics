import os
import argparse
import tqdm
import numpy as np
import multiprocessing as mp
import torch

from cs336_basics.utils import SimpleMapReduce
from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.bpe_trainer import train_bpe_impl
from cs336_basics.bpe_tokenizer import BPETokenizer
from cs336_basics.llm import Transformer
from cs336_basics.llm_train import (
    AdamW,
    get_batch,
    cross_entropy,
    save_checkpoint,
    gradient_clipping,
)

def train_tokenizer(
    tokenizer: str,
    vocab_size: int,
    input_path: str,
):
    mangled = tokenizer + "-" + str(vocab_size)
    vocab_path = os.path.join("assets", mangled, "vocab.json")
    merges_path = os.path.join("assets", mangled, "merges.txt")
    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        os.makedirs(os.path.join("assets", mangled), exist_ok=True)
        train_bpe_impl(
            input_path,
            vocab_size,
            ["<|endoftext|>"],
            os.path.join("assets", mangled)
        )

def write_back(tokens: list[int], dataset_path: str):
    size = os.path.getsize(dataset_path) if os.path.exists(dataset_path) else 0
    count = size // np.dtype(np.int64).itemsize
    new_count = count + len(tokens)
    new_size = new_count * np.dtype(np.int64).itemsize

    # 扩展文件
    with open(dataset_path, "ab") as f:
        f.truncate(new_size)

    # 重新映射写入
    mmap = np.memmap(dataset_path, dtype=np.int64, mode='r+', shape=(new_count,))
    mmap[count:new_count] = tokens
    mmap.flush()

def prepare_dataset_chunk(tokenizer:BPETokenizer, tmp_path:str, input_path:str, start:int, end:int):
    with open(input_path, mode="r") as f:
        token_count = 0
        f.seek(start)
        content = f.read(end - start)
        buffer = tokenizer.encode(content)
        write_back(buffer, tmp_path)

def prepare_dataset(
    tokenizer_name: str,
    input_path: str,
):
    dataset_dir = "assets/dataset"
    vocab_path = os.path.join("assets", tokenizer_name, "vocab.json")
    merges_path = os.path.join("assets", tokenizer_name, "merges.txt")
    os.makedirs(dataset_dir, exist_ok=True)
    mangled = os.path.split(input_path)[1].split(".")[0]
    mangled = tokenizer_name + "-" + mangled + ".dat"
    dataset_path = os.path.join(dataset_dir, mangled)

    tokenizer = BPETokenizer.from_files(vocab_path, merges_path,["<|endoftext|>"])

    tmp_path = os.path.join(dataset_dir,"tmp")
    os.makedirs(tmp_path, exist_ok=True)
    with open(input_path, mode='rb') as f:
        boundaries = find_chunk_boundaries(f, mp.cpu_count() * 4, b"<|endoftext|>")

    args = []
    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        tmp = os.path.join(tmp_path, str(i)+".tmp")
        args.append((tokenizer, tmp, input_path, start, end))

    with mp.Pool(processes=mp.cpu_count()//2) as pool:
        pool.starmap(prepare_dataset_chunk, args)

    for idx in range(len(boundaries) - 1):
        tmp = os.path.join(tmp_path, str(idx)+".tmp")
        tmp_count = os.path.getsize(tmp) // np.dtype(np.int64).itemsize
        tmp_mmap = np.memmap(tmp, dtype=np.int64, mode='r+', shape=(tmp_count,))
        write_back(tmp_mmap, dataset_path)

    # verify
    ds_count = os.path.getsize(dataset_path) // np.dtype(np.int64).itemsize
    ds_mmap = np.memmap(dataset_path, dtype=np.int64, mode='r+', shape=(ds_count,))
    print(tokenizer.decode(ds_mmap[-1000:]))

def train_model(
    dataset_path: str,
    checkpoint_dir: str,
    vocab_size: int,
    context_length: int,
    d_model: int,
    d_ff: int,
    rope_theta: int,
    n_layers: int,
    n_heads: int,
    batch_size: int,
    total_tokens: int,
):
    count = os.path.getsize(dataset_path) // np.dtype(np.int64).itemsize
    mmap = np.memmap(dataset_path, dtype=np.int64, mode='r', shape=(count,))
    device = "cuda:0"
    model = Transformer(
        vocab_size,
        context_length,
        d_model,
        n_layers,
        n_heads,
        d_ff,
        rope_theta,
        device=device
    )
    optimizer = AdamW(model.parameters())
    step = int(total_tokens / batch_size / context_length)
    max_norm = 1e-2
    os.makedirs(checkpoint_dir, exist_ok=True)
    for iter in range(step):
        x, y = get_batch(mmap, batch_size, context_length, device)
        optimizer.zero_grad()
        loss = cross_entropy(model(x), y)
        loss.backward()
        gradient_clipping(model.parameters(), max_norm)
        optimizer.step()
        print(iter, f"loss {loss.item():.2f} perplexity {torch.exp(loss.detach()).item():.2f}")
        if iter % 50 == 0:
            save_checkpoint(model, optimizer, iter, os.path.join(checkpoint_dir, str(iter)+".data"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Train tokenizer and model")
    subparsers = parser.add_subparsers(dest="command")

    parser_tokenizer = subparsers.add_parser("tokenizer", help="train tokenizer")
    parser_tokenizer.add_argument("-n", "--name", type=str, required=True, help="tokenizer's name")
    parser_tokenizer.add_argument("-s", "--size", type=int, required=True, help="vocab size")
    parser_tokenizer.add_argument("-i", "--input", type=str, required=True, help="train input")

    parser_dataset = subparsers.add_parser("dataset", help="prepare dataset")
    parser_dataset.add_argument("-t", "--tokenizer", type=str, required=True, help="tokenizer's name")
    parser_dataset.add_argument("-i", "--input", type=str, required=True, help="dataset src")

    parser_model = subparsers.add_parser("model", help="train model")
    parser_model.add_argument("--dataset", type=str, required=True, help="dataset path")
    parser_model.add_argument("--checkpoint", type=str, required=True, help="checkpoint path")
    parser_model.add_argument("--vocab", type=int, required=True, help="vocab size")
    parser_model.add_argument("--context", type=int, required=True, help="context length")
    parser_model.add_argument("--dmodel", type=int, required=True, help="d_model")
    parser_model.add_argument("--dff", type=int, required=True, help="d_ff")
    parser_model.add_argument("--rope", type=int, required=True, help="repe theta")
    parser_model.add_argument("--layers", type=int, required=True, help="n_layers")
    parser_model.add_argument("--heads", type=int,required=True, help="n_heads")
    parser_model.add_argument("--batch", type=int, required=True, help="batch size")
    parser_model.add_argument("--total", type=int, required=True, help="total tokens")

    args = parser.parse_args()
    if args.command == "tokenizer":
        train_tokenizer(
            args.name,
            args.size,
            args.input
        )
    elif args.command == "dataset":
        prepare_dataset(args.tokenizer, args.input)
    elif args.command == "model":
        train_model(
            args.dataset,
            args.checkpoint,
            args.vocab,
            args.context,
            args.dmodel,
            args.dff,
            args.rope,
            args.layers,
            args.heads,
            args.batch,
            args.total
        )