import re
import unicodedata

import jieba.posseg as pseg

from data import ndarray_to_tuple_array

with open('data/stopwords.txt', 'r', encoding='utf-8') as file:
    stop_words = file.read().split('\n')
stop_properties = []
stop_properties_general = list('efumoqy')

normalize_punctuations_table = {ord(f): ord(t) for f, t in zip(
    u'，。！？【】（）％＃＠＆１２３４５６７８９０',
    u',.!?[]()%#@&1234567890')}


def normalize_punctuations(str):
    return unicodedata.normalize('NFKC', str).translate(normalize_punctuations_table)


def desc_clean_clean(s):
    s = normalize_punctuations(s)
    s = re.sub(r'[0-9一二三四五六七八九][.、:]', ' ', s)
    s = re.sub(r'[-]{3,}|[*]{3,}', '', s)
    s = re.sub(r' ', '', s)
    return s


def filter_seg_result(pair):
    seg_word, properti = pair
    return (seg_word not in stop_words) and (properti not in stop_properties) \
           and (simplify_property(properti) not in stop_properties_general)


def simplify_property(p):
    if p == 'eng':
        return p

    return p[0]


def preprocess_input_with_properties(strs, split=False):
    def filter_words(x):
        seg_result = list(filter(filter_seg_result, pseg.cut(x)))
        return ''.join([w for w, p in seg_result])

    def clean_desc(s):
        s = desc_clean_clean(s)
        s = filter_words(s)
        if split:
            s = list(filter(lambda x: x != '', re.split(r'[.;!]', s)))
        return s

    strs = [clean_desc(s) for s in strs]

    return strs


def preprocess_input_w_prop_embeddings(strs, simplify_label=True):
    def filter_words(x):
        seg_result = list(filter(filter_seg_result, pseg.cut(x)))
        x = ''.join([w for w, p in seg_result])

        seg_result = a2g(seg_result)
        prop_labels = []

        new_x = ''

        seg_word = ''
        properti = ''
        begin = True
        for ch in x:
            if seg_word == '':
                seg_word, properti = next(seg_result)
                # simplify
                if simplify_label:
                    properti = simplify_property(properti)
                begin = True
            if begin:
                mark_prefix = 'B-'
                begin = False
            else:
                mark_prefix = 'I-'
            mark = mark_prefix + properti.upper()
            prop_labels.append(mark)
            seg_word = seg_word[1:]
            # new_x += ch.upper()
            new_x += ch

        x = new_x

        assert len(x) == len(prop_labels)

        return ndarray_to_tuple_array([list(x), prop_labels])

    def clean_desc(s):
        s = desc_clean_clean(s)
        s = filter_words(s)
        return s

    strs = [clean_desc(s) for s in strs]

    return strs


def a2g(x):
    return (n for n in x)
