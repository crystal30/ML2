# !/usr/bin/python
# -*- coding:utf-8 -*-

import sys
import jieba
import jieba.posseg


if __name__ == "__main__":
    # reload(sys)
    # sys.setdefaultencoding('utf-8')
    f = open('./data/novel.txt', encoding='utf-8')
    str = f.read()
    f.close()

    # seg = jieba.posseg.cut(str)
    # for word, flag in seg:
    #     print(word, flag, '|', end='')

    seg = jieba.cut(str)
    for word in seg:
        print(word, '|', end='')


