import os
import time

import numpy as np
import tensorflow as tf

from data import read_corpus, read_dictionary, random_embedding, tag2label
from extract_util import preprocess_input_with_properties
from model import BiLSTM_CRF
from utils import get_logger, get_entity

## Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1.0  # need ~700MB GPU memory


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


args = Map({'train_data': 'data_path',
            'test_data': 'data_path',
            'batch_size': 64,
            'epoch': 10,
            'hidden_dim': 300,
            'optimizer': 'Adam',
            'CRF': True,
            'lr': 0.001,
            'clip': 5.0,
            'dropout': 0.5,
            'update_embedding': True,
            'pretrain_embedding': 'random',
            'embedding_dim': 300,
            'shuffle': True,
            'mode': 'train',
            'demo_model': '1521112368',
            'window_size': 0,
            'strides': 1,
            'all_o_dropout': 0.9,
            'resume': 0,
            'prop2label': 'data_path'})

args.mode = 'train'
args.window_size = 11
args.epoch = 20
args.batch_size = 128
# args.demo_model = os.path.join('p', '1561963360')
# args.train_data = 'data_path_save/p'

## get char embeddings
word2id = read_dictionary(os.path.join('.', args.train_data, 'word2id.pkl'))
if args.pretrain_embedding == 'random':
    embeddings = random_embedding(word2id, args.embedding_dim)
else:
    embedding_path = 'pretrain_embedding.npy'
    embeddings = np.array(np.load(embedding_path), dtype='float32')

prop2id = read_dictionary(os.path.join('.', args.train_data, 'prop2id.pkl'))
prop_embeddings = random_embedding(prop2id, args.embedding_dim)

## read corpus and get training data
if args.mode != 'demo':
    train_path = os.path.join('.', args.train_data, 'train_data')
    test_path = os.path.join('.', args.test_data, 'test_data')
    train_data = read_corpus(train_path)
    test_data = read_corpus(test_path)
    test_size = len(test_data)

## paths setting
paths = {}
timestamp = str(int(time.time())) if args.mode == 'train' and args.resume <= 0 else args.demo_model
output_path = os.path.join('.', "data_path_save", timestamp)
if not os.path.exists(output_path): os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path): os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path
get_logger(log_path).info(str(args))

## training model
if args.mode == 'train':
    if args.resume > 0:
        ckpt_file = tf.train.latest_checkpoint(model_path)
        print(ckpt_file)
        paths['model_path'] = ckpt_file

    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config, prop_embeddings=prop_embeddings,
                       prop2label=prop2id)
    model.build_graph()

    ## hyperparameters-tuning, split train/dev
    # dev_data = train_data[:5000]; dev_size = len(dev_data)
    # train_data = train_data[5000:]; train_size = len(train_data)
    # print("train data: {0}\ndev data: {1}".format(train_size, dev_size))
    # model.train(train=train_data, dev=dev_data)

    ## train model on the whole training data
    print("train data: {}".format(len(train_data)))
    model.train(train=train_data, dev=test_data)  # use test_data as the dev_data to see overfitting phenomena

## testing model
elif args.mode == 'test':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config, prop_embeddings=prop_embeddings,
                       prop2label=prop2id)
    model.build_graph()
    print("test data: {}".format(test_size))
    model.test(test_data)

## demo
elif args.mode == 'demo':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config, prop_embeddings=prop_embeddings,
                       prop2label=prop2id)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)
        while 1:
            print('Please input your sentence (pre-process):')
            demo_sent = preprocess_input_with_properties([input()])[0]
            if demo_sent == '' or demo_sent.isspace():
                print('See you next time!')
                break
            else:
                demo_sent = list(demo_sent.strip())
                demo_data = [(demo_sent, ['O'] * len(demo_sent))]
                tag = model.demo_one(sess, demo_data)
                LBL = get_entity(tag, demo_sent)
                print('LBL: {}'.format(LBL))
                print('LBL(set): {}'.format(set(LBL)))
