import sys, pickle, os, random
import numpy as np

## tags, BIO

tag2label = {
    'O': 0, 'B-LBL': 1, 'I-LBL': 2
}


def read_corpus(corpus_path, w_prop_embedding=False):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        sent_, tag_ = [], []
        for line in fr.readlines():
            if line != '\n':
                spliced = line.strip().split()
                if w_prop_embedding:
                    [char, prop, label] = spliced
                    sent_.append((char, prop))
                else:
                    [char, label] = spliced
                    sent_.append(char)
                tag_.append(label)
            else:
                data.append((sent_, tag_))
                sent_, tag_ = [], []

    return data


# def vocab_build(vocab_path, corpus_path, min_count):
#     """
#
#     :param vocab_path:
#     :param corpus_path:
#     :param min_count:
#     :return:
#     """
#     data = read_corpus(corpus_path)
#     word2id = {}
#     for sent_, tag_ in data:
#         for word in sent_:
#             if word.isdigit():
#                 word = '<NUM>'
#             elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
#                 word = '<ENG>'
#             if word not in word2id:
#                 word2id[word] = [len(word2id) + 1, 1]
#             else:
#                 word2id[word][1] += 1
#     low_freq_words = []
#     for word, [word_id, word_freq] in word2id.items():
#         if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
#             low_freq_words.append(word)
#     for word in low_freq_words:
#         del word2id[word]
#
#     new_id = 1
#     for word in word2id.keys():
#         word2id[word] = new_id
#         new_id += 1
#     word2id['<UNK>'] = new_id
#     word2id['<PAD>'] = 0
#
#     print(len(word2id))
#     with open(vocab_path, 'wb') as fw:
#         pickle.dump(word2id, fw)


def sentence2id(sent, word2id, w_prop_embedding=False, prop2id=None):
    """

    :param prop2id:
    :param w_prop_embedding:
    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []

    def process_word(word):
        if word.isdigit():
            word = '<NUM>'
        # elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
        #     word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        return word

    def process_prop(prop):
        if prop not in prop2id:
            prop = '<UNK>'
        return prop

    if w_prop_embedding:
        for word, prop in sent:
            process_word(word)
            sentence_id.append((word2id[process_word(word)], prop2id[process_prop(prop)]))

    else:
        for word in sent:
            sentence_id.append(word2id[process_word(word)])

    return sentence_id


def read_dictionary(vocab_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0, max_len=None):
    """

    :param sequences:
    :param pad_mark:
    :param max_len:
    :return:
    """
    if max_len is None:
        max_len = max(map(lambda x: len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def pad_sequences_w_prop_embedding(sequences, pad_mark=0, max_len=None):
    """

    :param sequences:
    :param pad_mark:
    :param max_len:
    :return:
    """
    if max_len is None:
        max_len = max(map(lambda x: len(x), sequences))
    seq_list, props_list, seq_len_list = [], [], []
    for seq in sequences:
        [seq, props] = tuple_array_to_ndarray(seq)
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)

        props = list(props)
        props_ = props[:max_len] + [pad_mark] * max(max_len - len(props), 0)
        props_list.append(props_)

        seq_len_list.append(min(len(seq), max_len))

    return seq_list, props_list, seq_len_list


def to_sliding_window(sequences, window_size, all_O_dropout_rate, strides, labels=None, tag2label=None,
                      pad_mark=0):
    if tag2label is None:
        tag2label = {"O": 0}
    seqs_result = []
    seq_lens_result = []
    if labels is not None:
        negative_label = tag2label["O"]
        labels_result = []
        for seq, label in zip(sequences, labels):
            if (len(seq)) <= window_size:
                seq_list, _ = pad_sequences([seq], pad_mark=pad_mark, max_len=window_size)
                seqs_result += seq_list
                label_list, _ = pad_sequences([label], pad_mark=pad_mark, max_len=window_size)
                labels_result += label_list
                seq_lens_result.append(len(seq))
            else:
                seq_len = len(seq)
                i = 0
                while i + window_size < seq_len:
                    seq_window = seq[i:i + window_size]
                    label_window = label[i:i + window_size]
                    if not (all(v == negative_label for v in label_window)
                            and np.random.rand() <= all_O_dropout_rate):
                        seqs_result.append(seq_window)
                        labels_result.append(label_window)
                        seq_lens_result.append(window_size)
                    i += strides
                if i + window_size > seq_len:
                    seq_window = seq[seq_len - window_size:seq_len]
                    label_window = label[seq_len - window_size:seq_len]
                    if not (all(v == negative_label for v in label_window)
                            and np.random.rand() <= all_O_dropout_rate):
                        seqs_result.append(seq_window)
                        labels_result.append(label_window)
                        seq_lens_result.append(window_size)
        return seqs_result, labels_result, seq_lens_result
    else:
        for seq in sequences:
            if (len(seq)) <= window_size:
                seq_list, _ = pad_sequences([seq], pad_mark=pad_mark, max_len=window_size)
                seqs_result += seq_list
                seq_lens_result.append(len(seq))
            else:
                seq_len = len(seq)
                i = 0
                while i + window_size < seq_len:
                    seq_window = seq[i:i + window_size]
                    seqs_result.append(seq_window)
                    i += strides
                if i + window_size > seq_len:
                    seq_window = seq[seq_len - window_size:seq_len]
                    seqs_result.append(seq_window)
                    seq_lens_result.append(window_size)
        return seqs_result, None, seq_lens_result


def to_sliding_window_w_prop_embedding(sequences, window_size, all_O_dropout_rate, strides, labels=None, tag2label=None,
                                       pad_mark=0):
    if tag2label is None:
        tag2label = {"O": 0}
    seqs_result, props_result, seq_lens_result = [], [], []

    if labels is not None:
        negative_label = tag2label["O"]
        labels_result = []
        for seq, label in zip(sequences, labels):
            if (len(seq)) <= window_size:
                seq_list, prop_list, _ = pad_sequences_w_prop_embedding([seq], pad_mark=pad_mark, max_len=window_size)
                seqs_result += seq_list
                props_result += prop_list
                label_list, _ = pad_sequences([label], pad_mark=pad_mark, max_len=window_size)
                labels_result += label_list
                seq_lens_result.append(len(seq))
            else:
                seq_len = len(seq)
                i = 0
                while i + window_size < seq_len:
                    seq_window = seq[i:i + window_size]
                    label_window = label[i:i + window_size]
                    if not (all(v == negative_label for v in label_window)
                            and np.random.rand() <= all_O_dropout_rate):
                        [seq_window, prop_window] = tuple_array_to_ndarray(seq_window)
                        seqs_result.append(seq_window)
                        props_result.append(prop_window)
                        labels_result.append(label_window)
                        seq_lens_result.append(window_size)
                    i += strides
                if i + window_size > seq_len:
                    seq_window = seq[seq_len - window_size:seq_len]
                    label_window = label[seq_len - window_size:seq_len]
                    if not (all(v == negative_label for v in label_window)
                            and np.random.rand() <= all_O_dropout_rate):
                        [seq_window, prop_window] = tuple_array_to_ndarray(seq_window)
                        seqs_result.append(seq_window)
                        props_result.append(prop_window)
                        labels_result.append(label_window)
                        seq_lens_result.append(window_size)
        return seqs_result, props_result, labels_result, seq_lens_result
    else:
        for seq in sequences:
            if (len(seq)) <= window_size:
                seq_list, prop_list, _ = pad_sequences_w_prop_embedding([seq], pad_mark=pad_mark, max_len=window_size)
                seqs_result += seq_list
                props_result += prop_list
                seq_lens_result.append(len(seq))
            else:
                seq_len = len(seq)
                i = 0
                while i + window_size < seq_len:
                    seq_window = seq[i:i + window_size]
                    [seq_window, prop_window] = tuple_array_to_ndarray(seq_window)
                    seqs_result.append(seq_window)
                    props_result.append(prop_window)
                    i += strides
                if i + window_size > seq_len:
                    seq_window = seq[seq_len - window_size:seq_len]
                    [seq_window, prop_window] = tuple_array_to_ndarray(seq_window)
                    seqs_result.append(seq_window)
                    props_result.append(prop_window)
                    seq_lens_result.append(window_size)
        return seqs_result, props_result, None, seq_lens_result


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False, w_prop_embedding=False, prop2label=None):
    """

    :param prop2label:
    :param w_prop_embedding:
    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab, w_prop_embedding=w_prop_embedding, prop2id=prop2label)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels


def tuple_array_to_ndarray(tuple_array):
    return [list(tupl) for tupl in tuple_array_transpose(tuple_array)]


def ndarray_to_tuple_array(ndarray):
    return tuple_array_transpose(ndarray)


def tuple_array_transpose(m):
    return list(zip(*m))
