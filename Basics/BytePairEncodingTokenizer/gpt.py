import re
from collections import defaultdict

class BPETokenizer:
    def __init__(self, vocab_size):
        # vocab_size 表示词汇表最大容量，也就是最多允许合并多少次字对
        self.vocab_size = vocab_size
        self.bpe_vocab = {}  # 用来保存每次合并的字符对及其频率

    def _get_stats(self, corpus):
        # 统计语料中所有相邻字符对的频率
        # corpus 中每个元素形如 "l o w </w>"
        pairs = defaultdict(int)
        for word in corpus:
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += 1
        return pairs

    def _merge_vocab(self, pair, corpus):
        # 将出现频率最高的一对字符合并成一个新的子词
        pattern = re.escape(' '.join(pair))  
        replacement = ''.join(pair)
        new_corpus = [re.sub(pattern, replacement, w) for w in corpus]
        return new_corpus

    def train(self, word_list):
        """
        训练 BPE 模型：在每个单词末尾加 </w> 再拆成字符，
        重复找最频繁的字对并合并，直到达到 vocab_size。
        """
        # 关键修改：每个单词加 </w>，再拆为字符
        corpus = [' '.join(word) + ' </w>' for word in word_list]
        
        while len(self.bpe_vocab) < self.vocab_size:
            pairs = self._get_stats(corpus)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            corpus = self._merge_vocab(best_pair, corpus)
            # 记录最频繁的合并对
            self.bpe_vocab[best_pair] = pairs[best_pair]

    def tokenize(self, word):
        """
        用训练好的词表对新词做 BPE 分词。
        先给单词末尾加 </w>，依次替换所有合并对，最后去掉 </w>。
        """
        # 关键修改：分词时也加 </w>
        text = ' '.join(word) + ' </w>'
        
        # 按 (a+b) 的长度降序，一次性替换
        for (a, b) in sorted(self.bpe_vocab, key=lambda x: -len(''.join(x))):
            pattern = re.escape(f"{a} {b}")
            replacement = a + b
            text = re.sub(pattern, replacement, text)

        tokens = text.split()
        # 如果末尾是 </w>，移除之
        if tokens and tokens[-1] == '</w>':
            tokens.pop()
        return tokens

    def train_from_file(self, filepath):
        """
        从文本文件读取数据（每行有若干单词），
        拼成一个大列表后调用 train()
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        word_list = []
        for line in lines:
            word_list.extend(line.strip().split())
        self.train(word_list)

    def save_vocab(self, filepath):
        # 保存训练好的合并对
        with open(filepath, 'w', encoding='utf-8') as f:
            for (a, b), freq in self.bpe_vocab.items():
                f.write(f"{a} {b} {freq}\n")

    def load_vocab(self, filepath):
        # 从文件加载合并对
        self.bpe_vocab = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                a, b, freq = line.strip().split()
                self.bpe_vocab[(a, b)] = int(freq)

# -------------------- 测试主程序 --------------------

if __name__ == "__main__":
    tokenizer = BPETokenizer(vocab_size=19)  # 设置词汇表大小为 19

    # 从 data/bpe_test.txt 读取语料，并训练 BPE 模型
    tokenizer.train_from_file("data/bpe_test.txt")

    print("Tokenized output (vocab size = 19):\n")
    
    # 读取同一个文件，一行一行做分词并打印
    with open("data/bpe_test.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line_tokens = []
        for word in line.strip().split():
            line_tokens.extend(tokenizer.tokenize(word))
        print(" ".join(line_tokens))
