import os
import argparse
import tqdm
import numpy as np
from cs336_basics.bpe_trainer import train_bpe_impl
from cs336_basics.bpe_tokenizer import BPETokenizer
from cs336_basics.llm import Transformer
from cs336_basics.llm_train import AdamW, get_batch, cross_entropy, save_checkpoint

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
    with open(input_path, mode="r") as f:
        token_count = 0
        buffer = []
        for token in tokenizer.encode_iterable(f):
            buffer.append(token)
            if len(buffer) < 200000:
                continue
            token_count += len(buffer)
            write_back(buffer, dataset_path)
            buffer = []
            print(f"\rtoken count: {token_count}", end="")
        write_back(buffer, dataset_path)

def train_model(
    dataset_path: str,
    checkpoint_path: str,
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
    with open(dataset_path, mode='r') as f:
        f.seek(0, os.SEEK_END)
        count = f.tell() // np.dtype(np.int64).itemsize
    mmap = np.memmap(dataset_path, dtype=np.int64, mode='r', shape=(count,))
    model = Transformer(vocab_size, context_length, d_model, n_layers, n_heads, d_ff, rope_theta)
    optimizer = AdamW(model.parameters())
    step = int(total_tokens / batch_size / context_length)
    for iter in range(step):
        x, y = get_batch(mmap, batch_size, context_length)
        optimizer.zero_grad()
        loss = cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        save_checkpoint(model, optimizer, iter, checkpoint_path)

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
    parser_model.add_argument("-d", "--dataset", type=str, required=True, help="dataset path")

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
        raise NotImplementedError