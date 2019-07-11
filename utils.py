import argparse
import logging
import os
import string


def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_entity(tag_seq, char_seq):
    return get_LBL_entity(tag_seq, char_seq)


def get_LBL_entity(tag_seq, char_seq):
    length = len(char_seq)
    LBL = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-LBL':
            if 'lbl' in locals().keys():
                LBL.append(lbl)
                del lbl
            lbl = char
            if i + 1 == length:
                LBL.append(lbl)
        if tag == 'I-LBL':
            lbl += char
            if i + 1 == length:
                LBL.append(lbl)
        if tag not in ['I-LBL', 'B-LBL']:
            if 'lbl' in locals().keys():
                LBL.append(lbl)
                del lbl
            continue
    return LBL


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger


class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]


existing_labels = set()
if os.path.exists('existing_labels'):
    with open('existing_labels', mode='r', encoding='utf8') as f:
        existing_labels = eval(f.read())


def discovered_words(result):
    return result - existing_labels


def is_letter(c):
    return c in string.ascii_letters


def is_english(word):
    return all([is_letter(c) for c in word])
