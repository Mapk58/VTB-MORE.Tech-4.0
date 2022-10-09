import torch
from torch import nn
import transformers as T
import string
import re
from transformers import MBartTokenizer, MBartForConditionalGeneration
import pandas as pd
import numpy as np

class Summarizer:
    def __init__(self):
        self.model_name = "IlyaGusev/mbart_ru_sum_gazeta"
        self.tokenizer = MBartTokenizer.from_pretrained(self.model_name)
        self.model = MBartForConditionalGeneration.from_pretrained(self.model_name)
    
    def __call__(self, article_texts):
        #if not isinstance(article_texts, list):
        #    article_texts = [article_texts]

        input_ids = self.tokenizer([article_texts], max_length=600, truncation=True, return_tensors="pt",)["input_ids"]

        output_ids = self.model.generate(input_ids=input_ids, no_repeat_ngram_size=4)[0]
        summary = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return summary


class IndModel(nn.Module):
    def __init__(self, model_path="cointegrated/rubert-tiny", num_classes=25) -> None:
        super().__init__()

        self.tokenizer = T.AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
        self.model =T.AutoModel.from_pretrained("cointegrated/rubert-tiny")

        self.classifier = nn.Sequential(
            nn.Linear(in_features=312, out_features=num_classes),
            nn.Softmax(),
        )

    def forward(self, text):
        t = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        model_output = self.model(**{k: v.to(self.model.device) for k, v in t.items()})
        embeddings = model_output.last_hidden_state[:, 0, :]
        return self.classifier(embeddings)


class PrModel(nn.Module):
    def __init__(self, model_path="cointegrated/rubert-tiny", num_classes=1) -> None:
        super().__init__()

        self.tokenizer = T.AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
        self.model =T.AutoModel.from_pretrained("cointegrated/rubert-tiny")

        self.classifier = nn.Sequential(
            nn.Linear(in_features=312, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, text):
        t = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        model_output = self.model(**{k: v.to(self.model.device) for k, v in t.items()})
        embeddings = model_output.last_hidden_state[:, 0, :]
        return self.classifier(embeddings)


class Network():
    def __init__(self, task='industry') -> None:
        '''
        task может быть industry или profession

        industry - для вывода самых релевантных новостей в соответствии с интересующими пользователя отраслями
        profession - для определения рода деятельности человека, который заинтересуется данной новостью/новостями
        '''
        self.role2num = {
            'бухгалтер': 0,
            'генеральный директор': 1
        }
        self.idx2industry = {
        0: 'HR',
        1: 'Новые рынки',
        2: 'Документооборот',
        3: 'Инвестиции',
        4: 'Финтех',
        5: 'Финтех',
        6: 'Финтех',
        7: 'Бухгалтерия',
        8: 'Внешние рынки',
        9: 'Политика',
        10: 'Инвестиции',
        11: 'Юридические новости',
        12: 'Финансы',
        13: 'Бухгалтерия',
        14: 'Биотех',
        15: 'Инвестиции',
        16: 'Риски',
        17: 'Финансы',
        18: 'Инвестиции',
        19: 'Российский бизнес',
        20: 'Крупный бизнес',
        21: 'Риски',
        22: 'Энергетика',
        23: 'Логистика',
        24: 'Российский бизнес'
        }
        self.industry2idx = {val: key for key, val in self.idx2industry.items()}
        
        '''
        self.task = task
        if task == 'industry':
            self.model = IndModel()
            self.model.load_state_dict(torch.load('VTB-MORE.Tech-4.0/models/moretech/IndModel.pth'))
        else:
            self.model = PrModel()
            self.model.load_state_dict(torch.load('VTB-MORE.Tech-4.0/models/moretech/PrModel.pth'))
        '''
        self.summarizer = Summarizer()
    
    def bot_api(self, role: str, industries: list):
        answer = []

        role = self.role2num[role.lower()]
        industries = list(map(lambda x: self.industry2idx[x], industries))

        role_df = pd.read_csv('VTB-MORE.Tech-4.0/models/moretech/profession_labeled.csv')
        industry_df = pd.read_csv('VTB-MORE.Tech-4.0/models/moretech/industry_labeled.csv')

        role_idx = set(role_df[role_df['label'] == role].index)

        i = 0
        for industry in industries:
            try:
                industry_idx = list(set(industry_df[industry_df['label'] == industry].index).intersection(role_idx))[0]
            except:
                industry_idx = 5
        article_text = "Высота башни составляет 324 метра (1063 фута), примерно такая же высота, как у 81-этажного здания, и самое высокое сооружение в Париже. Его основание квадратно, размером 125 метров (410 футов) с любой стороны. Во время строительства Эйфелева башня превзошла монумент Вашингтона, став самым высоким искусственным сооружением в мире, и этот титул она удерживала в течение 41 года до завершения строительство здания Крайслер в Нью-Йорке в 1930 году. Это первое сооружение которое достигло высоты 300 метров. Из-за добавления вещательной антенны на вершине башни в 1957 году она сейчас выше здания Крайслер на 5,2 метра (17 футов). За исключением передатчиков, Эйфелева башня является второй самой высокой отдельно стоящей структурой во Франции после виадука Мийо."

        return self.summarizer(article_text)
    
    def label_new_data(self, text):
        pass


class BaseModel(nn.Module):
    def __init__(self, model_path="cointegrated/rubert-tiny", transformer_dropout=0.1, reinitialization_layers=0, pooling="mean"):
        super(BaseModel, self).__init__()
        
        self.config = T.AutoConfig.from_pretrained(model_path)
        self.config.hidden_dropout_prob = transformer_dropout
        self.config.hidden_dropout_prob = transformer_dropout
        self.config.dropout = transformer_dropout
        self.config.output_hidden_states = True
        self.pooling = pooling

        self.model = T.AutoModel.from_config(self.config)

    def _pool(self, hidden_state, strategy="cls"):
        strategy = strategy.lower()
        if strategy == "max":
            embeddings = torch.max(hidden_state, dim=1).values
        elif strategy == "mean":
            embeddings = torch.mean(hidden_state, dim=1)
        else:
            embeddings = hidden_state[:, 0, :]
            
        return embeddings

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0, std=self.config.initializer_range)

        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def _reinitializate_layers(self, layers, n=4):
        if n > 0:
            for layer in layers[-n:]:
                for module in layer.modules():
                    self._init_weights(module)
            else:
                print(f"Reinitializated Last {n} layers")


class ProfModel(BaseModel):
    def __init__(self, model_path="cointegrated/rubert-tiny", transformer_dropout=0.1, reinitialization_layers=0, pooling="mean"):
        super().__init__(model_path, transformer_dropout, reinitialization_layers, pooling)

        # классификатор рода деятельности (бухгалтер/руководитель)
        self.classifier_profession = nn.Sequential(
            nn.Linear(in_features=self.config.hidden_size, out_features=1),
            nn.Sigmoid(),
        )

        self._reinitializate_layers(self.model.encoder.layer, n=reinitialization_layers)
        self._init_weights(self.classifier_profession)

    def _reinitializate_layers(self, layers, n=4):
        return super()._reinitializate_layers(layers, n)
    
    def _init_weights(self, module):
        return super()._init_weights(module)
    
    def _pool(self, hidden_state, strategy="cls"):
        return super()._pool(hidden_state, strategy)
    
    def forward(self, description):
        description_input_ids, description_attention_mask = description
        
        description_output = self.model(input_ids=description_input_ids, 
                                        attention_mask=description_attention_mask)
        
        description_hidden_states = torch.stack(description_output.hidden_states, dim=0)
        description_hidden_state = description_hidden_states[-1]
        description_embeddings = self._pool(description_hidden_state, strategy=self.pooling)
        probability = self.classifier_profession(description_embeddings)
        
        return probability


class IndustryModel(BaseModel):
    def __init__(self, model_path="cointegrated/rubert-tiny", transformer_dropout=0.1, reinitialization_layers=0, pooling="mean", num_classes=10):
        super(IndustryModel, self).__init__(model_path, transformer_dropout, reinitialization_layers, pooling)

        self.classifier_industry = nn.Sequential(
            nn.Linear(in_features=self.config.hidden_size, out_features=num_classes),
            nn.Softmax(dim=1),
        )
        
        self._reinitializate_layers(self.model.encoder.layer, n=reinitialization_layers)
        self._init_weights(self.classifier_industry)
        
    def _reinitializate_layers(self, layers, n=4):
        return super()._reinitializate_layers(layers, n)
    
    def _init_weights(self, module):
        return super()._init_weights(module)
    
    def _pool(self, hidden_state, strategy="cls"):
        return super()._pool(hidden_state, strategy)
    
    def forward(self, description):
        description_input_ids, description_attention_mask = description
        
        description_output = self.model(input_ids=description_input_ids, 
                                        attention_mask=description_attention_mask)
        
        description_hidden_states = torch.stack(description_output.hidden_states, dim=0)
        description_hidden_state = description_hidden_states[-1]
        description_embeddings = self._pool(description_hidden_state, strategy=self.pooling)
        probabilities = self.classifier_industry(description_embeddings)
        
        return probabilities


class ClassificationPipeline:
    def __init__(self, model, tokenizer: T.PreTrainedTokenizer, device: str="cpu", task="industry", threshold=0.5):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.task = task
        self.threshold = threshold

    def tokenize(self, text, max_length=512):
        tokenized = self.tokenizer(text=text, 
                                   truncation=True, 
                                   padding="max_length", 
                                   max_length=max_length, 
                                   add_special_tokens=True, 
                                   return_attention_mask=True, 
                                   return_tensors="pt",
                                   return_token_type_ids=True)

        return (tokenized["input_ids"].to(self.device), tokenized["attention_mask"].to(self.device))
    
    def remove_punctuations(self, text):
        removed = text.translate(str.maketrans('', '', string.punctuation))
        return removed

    def remove_special_symbols(self, text):
        symbols = ["\n", "\r", "\xa0", "\u200b"]
        for symbol in symbols:
            text = text.replace(symbol, " ")

        return text
    
    def remove_extra_spaces(self, text):
        removed = re.sub(' +', ' ', text).strip()
        return removed
    
    def preprocess_description(self, text):
        preprocessed = text.strip().lower()
        # preprocessed = self.remove_html(preprocessed)
        preprocessed = self.remove_special_symbols(preprocessed)
        # preprocessed = remove_punctuations(preprocessed)
        preprocessed = self.remove_extra_spaces(preprocessed)
        
        return preprocessed

    def __call__(self, description):
        self.model.to(self.device)

        self.model.eval()
        with torch.no_grad():

            input_description = self.tokenize(description, max_length=512)
            input_description = self.preprocess_description(input_description)
            
            probabilities = self.model(description=input_description)
            if self.task.lower() == "industry":
                predictions = torch.argmax(probabilities.float().detach().to("cpu"), dim=1)
            else:
                proba = probabilities.detach().cpu().item()
                return 1 if proba >= self.threshold else 0
            
        return predictions


'''
tokenizer = T.AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
industry_model = IndustryModel("cointegrated/rubert-tiny2", num_classes=25)
industry_model_state = torch.load("models/industry_model.pth", map_location='cpu')
industry_model.load_state_dict(industry_model_state)

prof_model = ProfModel("cointegrated/rubert-tiny2")
prof_model_state = torch.load("models/prof_model.pth", map_location='cpu')
prof_model.load_state_dict(prof_model_state)


industry_pipeline = ClassificationPipeline(model=industry_model, tokenizer=tokenizer, device="cpu", task="industry")
prof_pipeline = ClassificationPipeline(model=prof_model, tokenizer=tokenizer, device="cpu", task="profession")

'''
