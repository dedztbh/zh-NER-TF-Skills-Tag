from main_core import main_core
from utils import Map

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
            'w_prop_embeddings': True
            })

args.mode = 'demo'
args.window_size = 0
args.epoch = 10
args.batch_size = 64
args.clip = 5.0
# args.dropout = 0.1
args.demo_model = 'h128'
args.hidden_dim = 128

main_core(args)
