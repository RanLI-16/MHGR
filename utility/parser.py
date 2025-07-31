import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="")

    #train
    parser.add_argument('--data_path', nargs='?', default='/home/dzh/Projects/create_hyper/multi-modal/recommended/', help='Input data path.')
    # parser.add_argument('--data_path', nargs='?', default='/home/dzh/Projects/Baseline/MICRO-main/data/', help='Input data path.')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--dataset', nargs='?', default='allrecipes', help='Choose a dataset from {sports, baby, microLens, tiktok, allrecipes, netflix}')
    parser.add_argument('--epoch', type=int, default=1000, help='Number of epoch.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size.')
    parser.add_argument('--embed_size', type=int, default=32,help='Embedding size.')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=3, help='GPU id')
    parser.add_argument('--Ks', nargs='?', default='[10, 20, 50]', help='K value of ndcg/recall @ k')
    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]', help='for emb_loss.')
    parser.add_argument('--lr', type=float, default=0.00045, help='Learning rate.')
    parser.add_argument('--weight_size', nargs='?', default='[64, 64]', help='Output sizes of every layer')
    parser.add_argument('--early_stopping_patience', type=int, default=9, help='')
    parser.add_argument('--layers', type=int, default=2, help='Number of feature graph conv layers')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1]', help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--sparse', type=int, default=1, help='Sparse or dense adjacency matrix')
    parser.add_argument('--test_flag', nargs='?', default='part', help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    # HGNN
    # parser.add_argument('--out_channels', type=float, default=32, help='out_channels')
    parser.add_argument('--drop_rate', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--id_cat_rate', type=float, default=0.36, help='before GNNs')
    parser.add_argument('--model_cat_rate', type=float, default=0.55, help='model_cat_rate')

    return parser.parse_args()


