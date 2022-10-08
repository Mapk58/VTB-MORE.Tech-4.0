import json
import requests
from bs4 import BeautifulSoup
import re
from concurrent.futures import ThreadPoolExecutor
from os import walk

user_agent = {
    'User-Agent': 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.63 Safari/537.36'}


def get_website_data(site):
    page = requests.get(site, headers=user_agent)
    soup = BeautifulSoup(page.content, "html.parser")
    return soup


def fix_link(link):
    link = link.replace(' ', '')
    try:
        if link[-1] in [';', ',']:
            link = link[:-1]
    except:
        # del raw_sites[j]
        return None
    if link[:4] not in ['http']:
        if link[:4] != 'www.':
            link = 'www.' + link
        if link[:4] != 'http':
            link = 'http://' + link
    return link

def get_links_from_page(path, page):
    starts = [m.start() + len('href="') for m in re.finditer('href="', page)]
    finishes = [(i, page.find('"', i)) for i in starts]
    raw_links = [page[i:j] for i, j in finishes]
    links = []
    bad_links = []
    full_links = []
    for link in raw_links:
        if len(link) > 4 and ((link[:4] != 'http' and link[-4:] in ['.htm', 'html']) or (
                '.' not in link and '(' not in link and ':' not in link and ';' not in link and '#' not in link[
                                                                                                           :3] and 'eng' not in link and '«' not in link and '№' not in link and '?' not in link)):
            links.append(link)
            if (link[0] != '/'):
                link = '/' + link
            if (path[-1] != '/'):
                full_links.append(path + link)
            else:
                full_links.append(path[:-1] + link)
            # full_links[-1] = full_links[-1][:full_links[-1].find('/', len(path)+1)+1]
        else:
            bad_links.append(link)
    return list(set(full_links))


def parse(site):
    data = {}
    try:
        data_from_site = get_website_data(site)
        first_layer = data_from_site.text.replace('\t', ' ').replace('\n\n', ' ').replace('  ', ' ')
        first_links = get_links_from_page(site, str(data_from_site))
        data = ({'link': site, 'first_layer': first_layer, 'first_links': first_links})
        return data
    except Exception as e:
        try:
            if (site[:5] == 'https'):
                data_from_site = get_website_data(site)
                first_layer = data_from_site.text.replace('\t', ' ').replace('\n\n', ' ').replace('  ', ' ')
                first_links = get_links_from_page(site, str(data_from_site))
                data = ({'link': site, 'first_layer': first_layer, 'first_links': first_links})
            else:
                # print('er ' + site + ' ' + filename)
                print(e)
                return
        except Exception as e:
            # print('er ' + site + ' ' + filename)
            print(e)
            return

def filter_links(link, links, d=3):
    return list(set([a.split('#')[0] for a in links if '/'.join(link.split('/')[:-d]) in a and len(link.split('/')) == len(a.split('/'))]))

def get_similar_links(link):

    base_link = '/'.join(link.split('/')[:3]) # i=3; добавить нормальное условие на поиск исходного сайта, если будет падать
    b = parse(fix_link(base_link))
    c = filter_links(link, b['first_links']) # надо поиграться с числом -3 и посмотреть, какие ссылки в каких ссылках мы ищем

    return c


if __name__ == '__main__':
    link = 'http://www.klerk.ru/blogs/Klerk_invest/537153/'
    # link = 'https://www.consultant.ru/legalnews/20541/'

    links = get_similar_links(link)

    for i in links:
        print(i)