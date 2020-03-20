import os
import json

from epub_conversion.utils import open_book, convert_epub_to_lines

booksDir = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__),'../books')))
dataDir = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__),'../data')))


books = []

for filename in os.listdir(booksDir):
    if filename.endswith('.epub'):
        books.append(filename)


import ebooklib
from ebooklib import epub

import re
def striphtml(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)

for filename in books:
    book = epub.read_epub(os.path.join(booksDir,filename))

    if os.path.exists(os.path.join(dataDir,filename.replace('.epub','') + ".json")):
        os.remove(os.path.join(dataDir,filename.replace('.epub','') + ".json"))

    with open(os.path.join(dataDir,filename.replace('.epub','') + ".json"), 'w', encoding="iso8859_2") as fd:
        
        my_dict = {}
        i = 1

        lista = [doc for doc in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)]

        for j in range(1,len(lista)):
            page = lista[j].content
        
            body_content = re.compile(r'<body.*?>(.*?)</body>', re.DOTALL).findall(page.decode("iso8859_2"))

        

            paragraph_list = re.compile(r'<p.*?>(.*?)</p>', re.DOTALL).findall(body_content[0])

            for k in range(len(paragraph_list)):
                my_dict[i] = striphtml(paragraph_list[k])
                i = i + 1
        
        json.dump(my_dict, fd, indent=4, ensure_ascii=False)

