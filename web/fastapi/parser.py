from getLinks import get_similar_links

def add_link_to_csv(link, roles):
    # link = 'http://www.klerk.ru/blogs/Klerk_invest/537153/'
    # roles = ['Генеральный директор', 'Общее']
    links = get_similar_links(link)
    print(links)

    # TODO:
    # add here some actions
    # with parsing and 
    # modifying csv
