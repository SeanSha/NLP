class BaseTokenizer:
    def tokenize(self, text:str) -> list[str]:
        raise NotImplementedError("Subclasses must implement this method.")

class IndoEuropeanTokenizer(BaseTokenizer):
    def __init__(self, lang='en'):
        self.lang = lang

    def tokenize(self, text: str) -> list[str]:
        import re
        pattern = r"[\wäöå']+|[^\w\s]"
        return re.findall(pattern, text)

class ChineseTokenizer(BaseTokenizer):
    def tokenize(self, text: str) -> list[str]:
        import jieba
        return list(jieba.cut(text))
    
def tokenize(text: str, lang: str = 'en') -> list[str]:
    if lang == 'zh':
        tokenizer = ChineseTokenizer()
    elif lang in ['en', 'sv']:
        tokenizer = IndoEuropeanTokenizer(lang = lang)
    else:
        raise ValueError(f"Unsupported language: {lang}")
    return tokenizer.tokenize(text)

if __name__ == "__main__":
    english_text = "I like apples, and I enjoy coding."
    chinese_text = "我喜欢苹果，也喜欢编程。"
    swedish_text = "Jag gillar äpplen och tycker om att koda."

    print("English tokens:", tokenize(english_text, lang='en'))
    print("Chinese tokens:", tokenize(chinese_text, lang='zh'))
    print("Swedish tokens:", tokenize(swedish_text, lang='sv'))    