import requests
from bs4 import BeautifulSoup

class URLBlock:

    def __init__(self):
        self.remove_patterns = []

    def _log(self, string):
      self.remove_patterns.append(string)

    def _get_full_text(self, url):
        soup = BeautifulSoup(requests.get(url).content, 'html.parser')
        return str(soup.body)

    def _all_contains(self, substring, array):
        return len(array) == len(list(filter(lambda x: substring in x, array)))

    def _remove_everywhere(self, substring, array):
        return [x.replace(substring, '') for x in array]

    def _recursive_tree_cleaner(self, node, array):
        if self._all_contains(str(node), array):
            self._log(str(node))
            return self._remove_everywhere(str(node), array)
            
        children = list(filter(lambda ch: ch.name is not None, node.children))
        for n in children:
            array = self._recursive_tree_cleaner(n, array)
        return array

    def transform(self, urls):
        contents = [self._get_full_text(url) for url in urls]
        for substring in self.remove_patterns:
            contents = self._remove_everywhere(substring, contents)
        return [BeautifulSoup(c, 'html.parser').text for c in contents]

    def fit_transform(self, urls):
        contents = [self._get_full_text(url) for url in urls]
        instance = BeautifulSoup(contents[0], 'html.parser')
        result = self._recursive_tree_cleaner(instance, contents)
        return [BeautifulSoup(r, 'html.parser').text for r in result]

    def getHeaders(self, urls):
        items = [BeautifulSoup(requests.get(url).content, 'html.parser').body for url in urls]
        return [item.find('h1').text for item in items]
        


# пример использования 

if __name__ == '__main__':

    urls = [
    'https://www.consultant.ru/legalnews/20528/',
    'https://www.consultant.ru/legalnews/20505/',
    'https://www.consultant.ru/legalnews/20533/',
    ]

    block = URLBlock()
    texts = block.fit_transform(urls)
    headers = block.getHeaders(urls)

    for header, text in zip(headers, texts):
        print('Заголовок:')
        print(header)
        print('Текст новости:')
        print(text.replace('\n', ' ').replace('    ',' ')[:1000]) # текста больше гораздо, [:1000] просто чтобы влезало в консоль
        print('-'*100)

    # test_texts = block.transform(test_urls)
    # test_headers = block.getHeaders(test_urls)

    # for header, text in zip(test_headers, test_texts):
    #     print(header)
    #     print(text.replace('\n', '')[:1000])
    #     print('-'*100)
