"""
提交位置
/data/nlp_course/assignments/hw1/{user}/
"""
# 需要提供.py 文件
import pickle
from typing import List, Tuple, Dict

from cs336_basics.bpe_tokenizer import BPETokenizer
from cs336_basics.bpe_trainer import BPE_Trainer

class BPE:
    def __init__(self):
        """
        初始化 BPE 模型。
        可在此定义：
        - self.vocab: 词汇表 (dict[str, int] 或 dict[int, str])
        - self.merges: 合并规则 (list[tuple[str, str]])
        - self.bpe_codes: 方便快速查找的合并映射
        """
        self.vocab = {}
        self.merges = []
        self.bpe_codes = {}

    def train(self, corpus: List[str], num_merges: int) -> None:
        """
        训练 BPE 模型：
        1. 将语料拆分为初始符号序列
        2. 统计符号对的出现频率
        3. 找出最频繁的一对并合并
        4. 重复 num_merges 次
        5. 保存最终的 merges 和 vocab

        参数：
            corpus: list[str]，训练语料，每个元素是一行文本
            num_merges: int，执行的合并次数
        """
        bpe_trainer = BPE_Trainer(None, num_merges + 256, [])
        bpe_trainer.train_with_pretoken(corpus)
        self.vocab = bpe_trainer.vocab
        self.merges = bpe_trainer.merges

    def encode(self, text: str) -> List[str]:
        """
        对输入文本进行分词，返回子词序列。

        参数：
            text: str, 输入文本
        返回：
            tokens: list[str], BPE 子词序列
        """
        bpe_tokenizer = BPETokenizer(self.vocab, self.merges, None)
        return bpe_tokenizer.encode(text)

    def decode(self, tokens: List[str]) -> str:
        """
        将子词序列还原为原始字符串。

        参数：
            tokens: list[str]
        返回：
            text: str
        """
        bpe_tokenizer = BPETokenizer(self.vocab, self.merges, None)
        return bpe_tokenizer.decode(tokens)

    def save(self, path_prefix: str) -> None:
        """
        保存 merges 与 vocab。
        输出：
            {path_prefix}_merges.pkl
            {path_prefix}_vocab.pkl
        """
        with open(f"{path_prefix}_merges.pkl", "wb") as f:
            pickle.dump(self.merges, f)
        with open(f"{path_prefix}_vocab.pkl", "wb") as f:
            pickle.dump(self.vocab, f)

    def load(self, path_prefix: str) -> None:
        """
        从 pkl 文件中加载 merges 与 vocab。
        输入：
            path_prefix: 文件名前缀（不含扩展名）
        """
        with open(f"{path_prefix}_merges.pkl", "rb") as f:
            self.merges = pickle.load(f)
        with open(f"{path_prefix}_vocab.pkl", "rb") as f:
            self.vocab = pickle.load(f)


if __name__ == "__main__":
    bpe = BPE()
    corpus = ["low", "lower", "lowest", "the", "quick", "brown", "fox"]
    bpe.train(corpus, num_merges=10)
    print(bpe.vocab)
    print(bpe.merges)
    sentences = [
        "low lower lowest",
        "newer wider",
        "the lower one"
    ]

    for s in sentences:
        tokens = bpe.encode(s)
        decoded = bpe.decode(tokens)

    bpe.save("student")
    bpe2 = BPE()
    bpe2.load("student")

    assert bpe2.vocab == bpe.vocab
    assert bpe2.merges == bpe.merges