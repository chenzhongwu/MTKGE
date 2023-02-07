import argparse
from utils import init_dir
from meta_trainer import MetaTrainer
import os
from subgraph import gen_subgraph_datasets
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='./icews05test_data.pkl')
    parser.add_argument('--state_dir', default='./state')
    parser.add_argument('--log_dir', default='./log')
    parser.add_argument('--tb_log_dir', default='./tb_log')

    parser.add_argument('--task_name', default='icews14')
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--num_exp', default=1, type=int)

    parser.add_argument('--train_bs', default=64, type=int)
    parser.add_argument('--eval_bs', default=16, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--num_step', default=100000, type=int)
    parser.add_argument('--log_per_step', default=10, type=int)
    parser.add_argument('--check_per_step', default=30, type=int)
    parser.add_argument('--early_stop_patience', default=20, type=int)
    parser.add_argument('--num_sample_cand', default=20, type=int)

    parser.add_argument('--dim', default=128, type=int)
    parser.add_argument('--ent_dim', default=None, type=int)
    parser.add_argument('--rel_dim', default=None, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--num_rel_bases', default=4, type=int)

    parser.add_argument('--kge', default='TeRo', type=str,
                        choices=['TComplEx', 'DistMult', 'ComplEx', 'RotatE','TTransE','T-DistMult','TComplEx','TeRo'])
    parser.add_argument('--metatrain_num_neg', default=32)
    parser.add_argument('--adv_temp', default=1, type=float)
    parser.add_argument('--gamma', default=10, type=float)
    parser.add_argument('--xishu', default=0.5, type=float)
    parser.add_argument('--reg', default=1, type=float)

    parser.add_argument('--cpu_num', default=10, type=float)
    parser.add_argument('--gpu', default='cuda:0', type=str)

    # subgraph
    parser.add_argument('--db_path', default=None)
    parser.add_argument('--num_train_subgraph', default=10000)
    parser.add_argument('--num_sample_for_estimate_size', default=10)
    parser.add_argument('--rw_0', default=10, type=int)
    parser.add_argument('--rw_1', default=10, type=int)
    parser.add_argument('--rw_2', default=5, type=int)

    # time
    parser.add_argument('--num_time', default=4017, type=int)
    parser.add_argument('--time_dim', default=128, type=int)

    args = parser.parse_args()


    args.db_path = args.data_path + '_subgraph'

    print(args.db_path)
    if not os.path.exists(args.db_path):
        gen_subgraph_datasets(args)

    init_dir(args)
    # adjust with KGE or TKGE modelling way
    args.ent_dim = args.dim
    args.rel_dim = args.dim
    args.time_dim = args.dim

    for run in range(args.num_exp):
        args.run = run
        args.exp_name = args.task_name + "_run" + str(args.run)

        trainer = MetaTrainer(args)
        trainer.train()

        del trainer
