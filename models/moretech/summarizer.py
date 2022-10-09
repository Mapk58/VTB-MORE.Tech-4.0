# !pip install transformers sentencepiece
from transformers import MBartTokenizer, MBartForConditionalGeneration

class Summarizer:
    def __init__(self) -> None:
        self.model_name = "IlyaGusev/mbart_ru_sum_gazeta"
        self.tokenizer = MBartTokenizer.from_pretrained(self.model_name)
        self.model = MBartForConditionalGeneration.from_pretrained(self.model_name)
    
    def __call__(self, article_texts):
        if type(article_texts) != "<class 'list'>":
            article_texts = [article_texts]

        input_ids = self.tokenizer(article_texts, max_length=2000, truncation=True, return_tensors="pt",)["input_ids"]

        output_ids = self.model.generate(input_ids=input_ids, no_repeat_ngram_size=4)[0]
        summary = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return summary
