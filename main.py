import argparse

from main_core import main_core
from utils import str2bool

## hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--train_data', type=str, default='data_path', help='train data source')
parser.add_argument('--test_data', type=str, default='data_path', help='test data source')
parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=40, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='random',
                    help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='train', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='1521112368', help='model for test and demo')

# sliding window settings
parser.add_argument('--window_size', type=int, default=0, help='window_size')
parser.add_argument('--strides', type=int, default=1, help='strides')
parser.add_argument('--all_o_dropout', type=float, default=0.9, help='all-o-dropout rate')

# resume settings
parser.add_argument('--resume', type=int, default=0, help='resume training @epoch, if already trained x, use x')

# custom tag2label
parser.add_argument('--custom_tag2label', type=str2bool, default=False, help='use custom tag2label')
parser.add_argument('--custom_tag2label_path', type=str, default='data_path', help='custom tag2label folder')

args = parser.parse_args()

main_core(args)
