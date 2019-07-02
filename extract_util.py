import re
import jieba.posseg as pseg

with open('data/stopwords.txt', 'r', encoding='utf-8') as file:
    stop_words = file.read().split('\n')
stop_properties = ['uj']


def valid_label(l):
    if len(l) <= 1:
        # print('invalid label removed:' + l)
        return False
    return True


def desc_clean_clean(s):
    s = re.sub(r'[0-9一二三四五六七八九][.、]', ' ', s)
    s = re.sub(r'[-]{3,}|[*]{3,}', '', s)
    s = re.sub(r' ', '', s)
    return s


def filter_seg_result(pair):
    seg_word, properti = pair
    return (seg_word not in stop_words) and (properti not in stop_properties)


sentence_break = list('。；！.;!')


def write_to_file(fileloc, z, every_sentence=True):
    sentences_record = []
    with open(fileloc, 'w+', encoding='utf-8') as f:
        prev_is_new_line = False
        for sentence, labels in z:
            sentence_record = ''
            for c, l in zip(sentence, labels):
                if c.strip() != '':
                    sentence_is_ending = every_sentence and c in sentence_break
                    if prev_is_new_line and sentence_is_ending:
                        prev_is_new_line = False
                    else:
                        f.write(c + ' ' + l + '\n')
                        sentence_record += c
                        if sentence_is_ending:
                            f.write('\n')
                            sentences_record.append(sentence_record)
                            sentence_record = ''
                            prev_is_new_line = True
                        else:
                            prev_is_new_line = False
            if not every_sentence:
                f.write('\n')
    return sentences_record


def preprocess_input_with_properties(strs, split=False):

    def filter_words(x):
        seg_result = list(filter(filter_seg_result, pseg.cut(x)))
        return ''.join([w for w, p in seg_result])

    def clean_desc(s):
        s = desc_clean_clean(s)
        s = filter_words(s)
        if split:
            s = list(filter(lambda x: x != '', re.split(r'[。；！.;!]', s)))
        return s

    strs = [clean_desc(s) for s in strs]

    return strs
