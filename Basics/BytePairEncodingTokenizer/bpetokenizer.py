import re
from collections import defaultdict

class BPETokenizer:
    def __init__(self, vocab_size):
        # vocab_size 表示词汇表最大容量，也就是最多允许合并多少次字对
        self.vocab_size = vocab_size
        self.bpe_vocab = {}  # 用来保存每次合并的字符对及其频率

    def _get_stats(self, corpus):
        # 统计语料中所有相邻字符对的频率（如 ('l', 'o') 出现了几次）
        pairs = defaultdict(int)
        for word in corpus:
            symbols = word.split()  # 把一个词按空格分成字符列表
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += 1
        return pairs

    def _merge_vocab(self, pair, corpus):
        # 将频率最高的一对字符合并成一个新的子词
        pattern = re.escape(' '.join(pair))  # 要匹配的字符对，比如 "l o"
        replacement = ''.join(pair)          # 合并后的字符，比如 "lo"
        new_corpus = [re.sub(pattern, replacement, word) for word in corpus]
        return new_corpus  # 返回更新后的语料

    def train(self, word_list):
        # 训练 BPE 模型：从单词列表中学习子词合并规则
        corpus = [' '.join(word) for word in word_list]  # 将每个单词拆成字符
        while len(self.bpe_vocab) < self.vocab_size:  # 循环直到达到设定的词表大小
            pairs = self._get_stats(corpus)  # 统计当前所有字对的频率
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)  # 找出出现最多的字对
            corpus = self._merge_vocab(best_pair, corpus)  # 合并它
            self.bpe_vocab[best_pair] = pairs[best_pair]   # 记录这次合并的字对及频率

    def tokenize(self, word):
        # 对一个词进行 BPE 分词（根据训练好的词表合并子词）
        tokens = ' '.join(word)  # 把单词拆成字符加空格，例如 "low" -> "l o w"
        for (a, b) in sorted(self.bpe_vocab, key=lambda x: -len(''.join(x))):
            # 遍历词表中记录的合并对，优先匹配更长的
            pattern = re.escape(f"{a} {b}")
            replacement = a + b
            tokens = re.sub(pattern, replacement, tokens)  # 替换字符对
        return tokens.split()  # 返回分词后的结果

    def train_from_file(self, filepath):
        # 从一个文本文件读取词语，并用来训练 BPE 模型
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        word_list = []
        for line in lines:
            word_list.extend(line.strip().split())  # 按空格分词
        self.train(word_list)

    def save_vocab(self, filepath):
        # 把训练好的 BPE 词表保存到文件中
        with open(filepath, 'w', encoding='utf-8') as f:
            for (a, b), freq in self.bpe_vocab.items():
                f.write(f"{a} {b} {freq}\n")

    def load_vocab(self, filepath):
        # 从文件中加载已有的 BPE 词表（避免重新训练）
        self.bpe_vocab = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                a, b, freq = line.strip().split()
                self.bpe_vocab[(a, b)] = int(freq)


# -------------------- 测试主程序 --------------------

if __name__ == "__main__":
    tokenizer = BPETokenizer(vocab_size=19)  # 设置词汇表大小为 19

    # 从文件中读取语料，训练 BPE 模型
    tokenizer.train_from_file("data/bpe_test.txt")

    # 再次打开文件，这次一行一行读取作为句子做分词
    with open("data/bpe_test.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    print("Tokenized output (vocab size = 19):\n")
    for line in lines:
        tokens = []
        for word in line.strip().split():
            # 对每个单词调用 tokenize() 进行 BPE 分词
            tokens.extend(tokenizer.tokenize(word))
        print(" ".join(tokens))  # 打印整行的分词结果
