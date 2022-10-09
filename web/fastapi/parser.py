from getLinks import get_similar_links
from parseNews import URLBlock
import csv 
import pandas as pd
import numpy as np

def add_link_to_csv(link, roles):
    # link = 'http://www.klerk.ru/blogs/Klerk_invest/537153/'
    # roles = ['Генеральный директор', 'Общее']
    links = get_similar_links(link)
    print(links)

    block = URLBlock()
    with open('./dataset.csv', 'a') as data_file:
        writer = csv.writer(data_file)
        writer.writerow(['title'])
        texts = block.fit_transform(links)
        for text in texts:
            text.replace('\n', ' ').replace('    ',' ')
            writer.writerow([text.replace('\n', ' ').replace('    ',' ')])
    
    df = pd.read_csv('./dataset.csv')
    len = len(df.index)
    index = np.arange(len)
    df.insert(0, column='index', value=index)
    df.set_index('index', drop=True, inplace=True)
    df.to_csv('./new_dataset.csv')