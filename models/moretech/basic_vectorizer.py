import pickle
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
import pymorphy2
from itertools import chain


class TextVectorizer:
    idx2industry = {
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

    industry2idx = {val: key for key, val in idx2industry.items()}
    
    def __init__(self, vectorizer_path=None, model_path=None, n_components=25) -> None:

        nltk.download('punkt')

        self.word_tokenizer = RegexpTokenizer(r'[a-zа-яёЁА-ЯA-Z]+|[^\w\s]|\d+')
        self.sent_tokenizer = lambda sent: nltk.sent_tokenize(sent, language="russian")
        nltk.download('stopwords')
        self.morph = pymorphy2.MorphAnalyzer()
        self.stops = nltk.corpus.stopwords.words('russian')

        if vectorizer_path is not None and model_path is not None:
            with open(vectorizer_path, 'rb') as f:
                self.count = pickle.load(f)

            with open(model_path, 'rb') as f:
                self.topic_model = pickle.load(f)
        else:
            self.count = CountVectorizer(binary=True)
            self.topic_model = LatentDirichletAllocation(
                n_components=n_components,
                max_iter=5,
                learning_method="online",
                learning_offset=50.0,
                random_state=0,
                )
    
    def plot_top_words(self, n_top_words, title):
        feature_names = self.count.get_feature_names_out()
        fig, axes = plt.subplots(5, 5, figsize=(60, 30), sharex=True)
        axes = axes.flatten()
        for topic_idx, topic in enumerate(self.topic_model.components_):
            top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
            top_features = [feature_names[i] for i in top_features_ind]
            weights = topic[top_features_ind]

            ax = axes[topic_idx]
            ax.barh(top_features, weights, height=0.7)
            ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 30})
            ax.invert_yaxis()
            ax.tick_params(axis="both", which="major", labelsize=20)
            for i in "top right left".split():
                ax.spines[i].set_visible(False)
            fig.suptitle(title, fontsize=40)

        plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
        plt.show()
    
    def proccess_text(self, text):
        text = text.lower()
        sents = self.sent_tokenizer(text)
        words = list(chain.from_iterable(self.word_tokenizer.tokenize_sents(sents)))

        return [x for x in [self.morph.normal_forms(word)[0] for word in words] if x not in self.stops]
    
    def fit(self, data):
        '''
        data is a raw text
        '''
        bag = self.count.fit_transform(data)
        self.topic_model.fit(bag)
    
    def transform(self, data):
        return self.topic_model.transform(self.count.transform(data))
    
    def save(self, vectorizer_path, model_path):
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.count, f)
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.topic_model, f)
    
    def make_labeled_dataset(self, data):
        bag = self.count.transform(data)
        topics = np.argmax(self.topic_model.transform(bag), axis=1)

        return [(text, label) for text, label in zip(data, topics)]

