from collections import defaultdict
import math
import tokenizer as token

class NaiveBayesTextClassification:
    def __init__(self, path):
        self.path = path
        self.word_counts = {
            0: defaultdict(int),  # 负面词频统计
            1: defaultdict(int)   # 正面词频统计
        } # 用来统计每个类别中每个单词的频率

        self.total_words = { 0: 0, 1: 0 } # 用来统计每个类别的总单词数
        self.doc_counts = { 0: 0, 1: 0 } # 用来统计每个类别的评论数
        self.vocab = set() # 用来保存所有的单词，用于平滑
        self.build_bag_of_words(path)

    def build_bag_of_words(self, path):
        # 构建 bag of words
        for sentence, label in load_from_tsv(path):
            self.doc_counts[label] += 1 # 按照正面或负面评论统计+1
            words = token.tokenize(sentence, 'en')
            words = self.handle_negation(words, window_size=3)
            for word in set(words):
                self.word_counts[label][word] += 1 # 按照正面或负面标签,统计出现的单词次数
                self.total_words[label] += 1 # 按照正负评论,统计在正负评论里分别出现的单词的总的次数,包括重复的
                self.vocab.add(word) # 讲单词存到词汇表

    """
    这部分公式用来计算先验概率
    P(正面) = 正面评论数 / (正面评论数 + 负面评论数)
    P(负面) = 负面评论数 / (正面评论数 + 负面评论数)
    # 使用标准注释实现折叠
    # 这是正面情感的先验概率公式：
    # P(正面) = self.doc_counts[1] / (self.doc_counts[0] + self.doc_counts[1])
    # 这是负面情感的先验概率公式：
    # P(负面) = self.doc_counts[0] / (self.doc_counts[0] + self.doc_counts[1])
    """
    def calculate_priorprobability(self):
        total_comments = self.doc_counts[0] + self.doc_counts[1]
        self.prior_positive = self.doc_counts[1] / total_comments
        self.prior_negative = self.doc_counts[0] / total_comments
        return self.prior_positive, self.prior_negative

    """
    这部分公式用来计算条件概率
    P(w_i | C) = 该类别 C 中单词 w_i 的出现次数 + 1 / 该类别 C 中所有单词的总数 + 词汇表大小
    # 使用拉普拉斯平滑：避免单词在某类别下没有出现时其概率为零
    # self.word_counts[label][word] 表示单词 word 在类别 label 下的出现次数
    # self.total_words[label] 表示类别 label 下所有单词的总数
    # len(self.vocab) 表示词汇表大小，包含所有不同的单词
    """
    def calculate_conditionalprobability(self, word):

        prob_word_postive = ( self.word_counts[1][word] + 1) / (sum(self.word_counts[1].values()) + len(self.vocab))
        prob_word_negative = ( self.word_counts[0][word] + 1) / (sum(self.word_counts[0].values()) + len(self.vocab))
        
        return prob_word_postive, prob_word_negative

    """
    这部分用来计算预测结果
    P(C | w_1, w_2, ..., w_n) ∝ P(C) * P(w_1 | C) * P(w_2 | C) * ... * P(w_n | C)
    # P(C) 是先验概率，表示类别 C 的概率
    # P(w_i | C) 是条件概率，表示在类别 C 下单词 w_i 出现的概率
    # 对每个测试句子，我们计算该句子属于正面或负面类别的概率，选择概率较大的类别作为预测结果
    """
    def predict(self, sentence):

        prior_positive, prior_negative = self.calculate_priorprobability()

        log_prob_postive = math.log(prior_positive)
        log_prob_negative = math.log(prior_negative)
        
        for word in token.tokenize(sentence, 'en'):
            prob_word_postive, prob_word_negative = self.calculate_conditionalprobability(word)

            log_prob_postive += math.log(prob_word_postive)
            log_prob_negative += math.log(prob_word_negative)
        
        return 1 if log_prob_postive > log_prob_negative else 0
    
    def handle_negation(self, tokens, window_size=3):
        NEGATION_WORDS = {"no", "not", "never", "n't", "cannot"}
        result = []
        i = 0
        while i < len(tokens):
            word = tokens[i]
            if word.lower() in NEGATION_WORDS:
                result.append(word)  # 否定词本身保留
                j = 1
                while j <= window_size and (i + j) < len(tokens):
                    if tokens[i + j] in {".", ",", "!", "?", ";", ":"}:
                        break
                    neg_word = "NOT_" + tokens[i + j]
                    result.append(neg_word)
                    j += 1
                i = i + j
            else:
                result.append(word)
                i += 1
        return result

def calculate_accuracy(predit_list):

        total_count = len(predit_list)
        correct_count = predit_list.count(1)
        return (correct_count / total_count) * 100


def load_from_tsv(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if '\t' not in line:
                continue
            text, label = line.strip().rsplit('\t', 1)
            yield text, int(label)


if __name__ == "__main__":

    import os

    base_path = "sentiment_sentences"  # 修改1：设置到上一级目录
    sections = ["imdb", "amazon", "yelp"]  # 修改2：设置要测试的三个数据集

    for section in sections:  # 修改3：循环每个数据集
        train_data_path = os.path.join(base_path, "train", f"{section}-train.tsv")
        test_data_path = os.path.join(base_path, "test", f"{section}-test.tsv")

        predict_test_results = []
        actual_test_data_labels = []
        accuracy_list = []

        bayes_model = NaiveBayesTextClassification(train_data_path)

        for test_sentence, label in load_from_tsv(test_data_path):
            actual_test_data_labels.append(label)
            predict_result = bayes_model.predict(test_sentence)
            predict_test_results.append(predict_result)
            accuracy_list.append(1 if predict_result == label else 0)

        accurate_precentage = calculate_accuracy(accuracy_list)
        print(f"{section.capitalize()} accurate prediction results is {accurate_precentage}%")

