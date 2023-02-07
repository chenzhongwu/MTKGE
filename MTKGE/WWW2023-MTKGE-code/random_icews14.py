import pickle
import numpy as np
from utils import get_g, serialize
import torch
import lmdb
import dgl
from collections import defaultdict as ddict
from tqdm import tqdm
import random
from scipy import sparse
import multiprocessing as mp
import numpy as np
from itertools import chain
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--s1', default=100)
parser.add_argument('--l1', default=10)
parser.add_argument('--entity_r1', default=0.7)
parser.add_argument('--relation_r1', default=0)
parser.add_argument('--s2', default=100, type=int)
parser.add_argument('--l2', default=10, type=int)
parser.add_argument('--s3', default=100, type=int)
parser.add_argument('--l3', default=10, type=int)
# parser.add_argument('--a1', default=0.5, type=int)
args = parser.parse_args()
data_train = open('./data/icews14/train.txt', 'r', encoding='utf-8')
data_valid = open('./data/icews14/valid.txt', 'r', encoding='utf-8')
data_test = open('./data/icews14/test.txt', 'r', encoding='utf-8')
# entity_2id = open('./data/icews14/entity2id.txt', 'r', encoding='utf-8')
# relation_2id = open('./data/icews14/relation2id.txt', 'r', encoding='utf-8')
entity_num = 7128
relation_num = 230
time_num = 365
def read_entity_2id():
    entity_2id={}
    id2_entity={}
    with open('./data/icews14/entity2id.txt') as f:
        for line in f.readlines():
            entity, id = line.strip().split('\t')
            entity_2id[entity] = int(id)
            id2_entity[int(id)] = entity
    return entity_2id, id2_entity
def read_relation_2id():
    relation_2id={}
    id_2relation={}
    with open('./data/icews14/relation2id.txt') as f:
        for line in f.readlines():
            relation, id = line.strip().split('\t')
            relation_2id[relation] = int(id)
            id_2relation[int(id)] = relation
    return relation_2id, id_2relation
def read_time_2id():
    time_2id={}
    id_2time={}
    with open('./data/icews14/time2id.txt') as f:
        for line in f.readlines():
            time, id = line.strip().split('\t')
            time_2id[time] = int(id)
            id_2time[int(id)] = time
    return time_2id, id_2time
def build_icews_data():
    data = chain(data_train)
    all_entity_2id, all_id2_entity = read_entity_2id()
    all_relation_2id, all_id2_relation = read_relation_2id()
    all_time_2id, all_id2_time = read_time_2id()
    src_list = []
    dis_list = []
    rel_list = []
    time_list = []
    for i, line in enumerate(data):
        head, relation, tail, time = line.strip().split('\t')
        src_list.append(all_entity_2id[head])
        dis_list.append(all_entity_2id[tail])
        rel_list.append(all_relation_2id[relation])
        time_list.append(all_time_2id[time])
    KG = dgl.graph((src_list, dis_list))
    KG.edata['e_id_zhen'] = torch.tensor(list(np.arange(len(rel_list))))
    KG.edata['relation_zhen'] = torch.tensor(rel_list)
    KG.edata['time_zhen'] = torch.tensor(time_list)
    KG.ndata['ent_id_zhen'] = torch.tensor(list(np.arange(entity_num)))

    KG_yuan = dgl.graph((src_list, dis_list))
    KG_yuan.edata['e_id_zhen'] = torch.tensor(list(np.arange(len(rel_list))))
    KG_yuan.edata['relation_zhen'] = torch.tensor(rel_list)
    KG_yuan.edata['time_zhen'] = torch.tensor(time_list)
    KG_yuan.ndata['ent_id_zhen'] = torch.tensor(list(np.arange(entity_num)))

    ent_id_list = list(np.arange(entity_num))
    test_E1 = random.sample(ent_id_list, args.s1)

    test_E1_entity_id, test_E1_egde_id, _ = dgl.sampling.random_walk(KG, test_E1, length=args.l1, return_eids=True)
    test_E1_total_entity_id, test_E1_total_entity_id_count = torch.unique(test_E1_entity_id, return_counts=True)
    E2_entity_id = list(torch.unique(test_E1_total_entity_id)) if -1 not in test_E1_total_entity_id else list(
        torch.unique(test_E1_total_entity_id))[1:]
    KG_test = dgl.node_subgraph(KG_yuan, E2_entity_id)

    remove_eid_list = []
    for i in range(KG_test.num_edges()):
        eid = KG_test.edata['e_id_zhen'][i]
        remove_eid_list.append(eid)

    ent_id_list = list(np.arange(entity_num))

    valid_E3 = random.sample(ent_id_list, args.s2)
    valid_E3_entity_id, valid_E3_egde_id, _ = dgl.sampling.random_walk(KG_yuan, valid_E3, length=args.l2, return_eids=True)
    valid_E3_total_entity_id, valid_E3_total_entity_id_count = torch.unique(valid_E3_entity_id, return_counts=True)
    E4_entity_id = list(torch.unique(valid_E3_total_entity_id)) if -1 not in valid_E3_total_entity_id else list(
        torch.unique(valid_E3_total_entity_id))[1:]

    KG_valid = dgl.node_subgraph(KG_yuan, E4_entity_id)

    for i in range(KG_valid.num_edges()):
        eid = KG_valid.edata['e_id_zhen'][i]
        remove_eid_list.append(eid)
    KG.remove_edges(remove_eid_list)


    rel_id_list = list(np.arange(relation_num))
    ent_id_list = list(np.arange(entity_num))
    rel_remove_id = random.sample(rel_id_list, int(args.relation_r1 * relation_num))
    ent_remove_id = random.sample(ent_id_list, int(args.entity_r1 * entity_num))
    KG.remove_nodes(ent_remove_id)
    src_KG, dis_KG = KG.edges()
    remove_edgs_list = []
    for i in range(KG.num_edges()):
        if KG.edata['relation_zhen'][i] in rel_remove_id:
            remove_edgs_list.append(i)
    KG.remove_edges(remove_edgs_list)

    # train
    remain_ent_type = KG.num_nodes()
    rel_ent_list = list(np.arange(remain_ent_type))
    train_ent_sample = random.sample(rel_ent_list, args.s3)

    train_E5_entity_id, train_egde_id, _ = dgl.sampling.random_walk(KG, train_ent_sample, length=args.l2, return_eids=True)
    train_total_entity_id, train_total_entity_id_count = torch.unique(train_E5_entity_id, return_counts=True)
    train_entity_id = list(torch.unique(train_total_entity_id)) if -1 not in train_total_entity_id else list(
        torch.unique(train_total_entity_id))[1:]
    KG_train = dgl.node_subgraph(KG, train_entity_id)
    # print(KG_train.ndata['ent_id_zhen'])
    # print(KG_test.ndata['ent_id_zhen'])
    # print(KG_valid.ndata['ent_id_zhen'])

    icews_data = dict()
    icews_data['train'] = dict()
    icews_data['train']['ent2id'] = dict()
    icews_data['train']['rel2id'] = dict()
    icews_data['train']['time2id'] = dict()
    icews_data['train']['quater'] = []

    icews_data['test'] = dict()
    icews_data['test']['ent2id'] = dict()
    icews_data['test']['rel2id'] = dict()
    icews_data['test']['time2id'] = dict()
    icews_data['test']['ent_map_list'] = []
    icews_data['test']['support'] = []
    icews_data['test']['rel_map_list'] = []
    icews_data['test']['time_map_list'] = []
    icews_data['test']['query_uent'] = []
    icews_data['test']['query_urel'] = []
    icews_data['test']['query_uboth'] = []
    icews_data['test']['query_utime'] = []

    icews_data['valid'] = dict()
    icews_data['valid']['ent2id'] = dict()
    icews_data['valid']['rel2id'] = dict()
    icews_data['valid']['time2id'] = dict()
    icews_data['valid']['ent_map_list'] = []
    icews_data['valid']['time_map_list'] = []
    icews_data['valid']['rel_map_list'] = []
    icews_data['valid']['support'] = []
    icews_data['valid']['query'] = []

    for i in range(KG_train.num_nodes()):
        icews_data['train']['ent2id'][all_id2_entity[int(KG_train.ndata['ent_id_zhen'][i])]] = i
    count = 0
    for rela in range(KG_train.num_edges()):
        if all_id2_relation[int(KG_train.edata['relation_zhen'][rela])] not in icews_data['train']['rel2id']:
            icews_data['train']['rel2id'][all_id2_relation[int(KG_train.edata['relation_zhen'][rela])]] = count
            count += 1
    CC = 0
    for T in range(KG_train.num_edges()):
        if all_id2_time[int(KG_train.edata['time_zhen'][T])] not in icews_data['train']['time2id']:
            icews_data['train']['time2id'][all_id2_time[int(KG_train.edata['time_zhen'][T])]] = CC
            CC += 1
    src_list, dis_list = KG_train.edges()
    for i in range(KG_train.num_edges()):
        icews_data['train']['quater'].append((int(src_list[i]),
                                              icews_data['train']['rel2id'][all_id2_relation[int(KG_train.edata['relation_zhen'][i])]],
                                              int(dis_list[i]),
                                              icews_data['train']['time2id'][all_id2_time[int(KG_train.edata['time_zhen'][i])]]
                                              ))

    for i in range(KG_test.num_nodes()):
        icews_data['test']['ent2id'][all_id2_entity[int(KG_test.ndata['ent_id_zhen'][i])]] = i
    count = 0
    for rela in range(KG_test.num_edges()):
        if all_id2_relation[int(KG_test.edata['relation_zhen'][rela])] not in icews_data['test']['rel2id']:
            icews_data['test']['rel2id'][all_id2_relation[int(KG_test.edata['relation_zhen'][rela])]] = count
            count += 1
    CC = 0
    for T in range(KG_test.num_edges()):
        if all_id2_time[int(KG_test.edata['time_zhen'][T])] not in icews_data['test']['time2id']:
            icews_data['test']['time2id'][all_id2_time[int(KG_test.edata['time_zhen'][T])]] = CC
            CC += 1
    for k, v in icews_data['test']['ent2id'].items():
        if k in icews_data['train']['ent2id']:
            icews_data['test']['ent_map_list'].append(icews_data['train']['ent2id'][k])
        else:
            icews_data['test']['ent_map_list'].append(-1)

    for k, v in icews_data['test']['rel2id'].items():
        if k in icews_data['train']['rel2id']:
            icews_data['test']['rel_map_list'].append(icews_data['train']['rel2id'][k])
        else:
            icews_data['test']['rel_map_list'].append(-1)

    for k, v in icews_data['test']['time2id'].items():
        if k in icews_data['train']['time2id']:
            icews_data['test']['time_map_list'].append(icews_data['train']['time2id'][k])
        else:
            icews_data['test']['time_map_list'].append(-1)

    src_list, dis_list = KG_test.edges()
    rel_list = KG_test.edata['relation_zhen']
    time_list = KG_test.edata['time_zhen']
    for i in range(KG_test.num_edges()):

        icews_data['test']['support'].append((int(src_list[i]),
                                                  icews_data['test']['rel2id'][all_id2_relation[int(rel_list[i])]],
                                                  int(dis_list[i]),
                                                  icews_data['test']['time2id'][   all_id2_time[int(time_list[i])]   ]))
        if bool(1 - (icews_data['test']['ent_map_list'][int(src_list[i])] != -1 and
                     icews_data['test']['ent_map_list'][int(dis_list[i])] != -1)) \
                and icews_data['test']['rel_map_list'][icews_data['test']['rel2id'][ all_id2_relation[int(rel_list[i])]   ]] != -1 and \
                icews_data['test']['time_map_list'][icews_data['test']['time2id'][  all_id2_time[int(time_list[i])]   ]] != -1:
            icews_data['test']['query_uent'].append((int(src_list[i]),
                                                     icews_data['test']['rel2id'][   all_id2_relation[int(rel_list[i])]   ],
                                                     int(dis_list[i]),
                                                     icews_data['test']['time2id'][all_id2_time[int(time_list[i])]]))
        if icews_data['test']['ent_map_list'][int(src_list[i])] != -1 and \
                icews_data['test']['ent_map_list'][int(dis_list[i])] != -1 \
                and icews_data['test']['rel_map_list'][icews_data['test']['rel2id'][  all_id2_relation[int(rel_list[i])] ]] == -1 and \
                icews_data['test']['time_map_list'][icews_data['test']['time2id'][all_id2_time[int(time_list[i])]]] != -1:
            icews_data['test']['query_urel'].append((int(src_list[i]),
                                                     icews_data['test']['rel2id'][ all_id2_relation[int(rel_list[i])]   ],
                                                     int(dis_list[i]),
                                                     icews_data['test']['time2id'][  all_id2_time[int(time_list[i])]]))
        if icews_data['test']['ent_map_list'][int(src_list[i])] == -1 and \
                icews_data['test']['ent_map_list'][int(dis_list[i])] == -1 \
                and icews_data['test']['rel_map_list'][icews_data['test']['rel2id'][   all_id2_relation[int(rel_list[i])]  ]] == -1 and \
                icews_data['test']['time_map_list'][icews_data['test']['time2id'][   all_id2_time[int(time_list[i])] ]] != -1:
            icews_data['test']['query_uboth'].append((int(src_list[i]),
                                                      icews_data['test']['rel2id'][  all_id2_relation[int(rel_list[i])]   ],
                                                      int(dis_list[i]),
                                                      icews_data['test']['time2id'][  all_id2_time[int(time_list[i])]]))
        if icews_data['test']['ent_map_list'][int(src_list[i])] != -1 and \
                icews_data['test']['ent_map_list'][int(dis_list[i])] != -1 \
                and icews_data['test']['rel_map_list'][icews_data['test']['rel2id'][   all_id2_relation[int(rel_list[i])]  ]] != -1 and \
                icews_data['test']['time_map_list'][icews_data['test']['time2id'][all_id2_time[int(time_list[i])]]] == -1:
            icews_data['test']['query_utime'].append((int(src_list[i]),
                                                      icews_data['test']['rel2id'][   all_id2_relation[int(rel_list[i])]   ],
                                                      int(dis_list[i]),
                                                      icews_data['test']['time2id'][all_id2_time[int(time_list[i])]]))

    for i in range(KG_valid.num_nodes()):
        icews_data['valid']['ent2id'][all_id2_entity[int(KG_valid.ndata['ent_id_zhen'][i])]] = i
    count = 0
    for rela in range(KG_valid.num_edges()):
        if all_id2_relation[int(KG_valid.edata['relation_zhen'][rela])] not in icews_data['valid']['rel2id']:
            icews_data['valid']['rel2id'][all_id2_relation[int(KG_valid.edata['relation_zhen'][rela])]] = count
            count += 1
    CC = 0
    for T in range(KG_valid.num_edges()):
        if all_id2_time[int(KG_valid.edata['time_zhen'][T])] not in icews_data['valid']['time2id']:
            icews_data['valid']['time2id'][all_id2_time[int(KG_valid.edata['time_zhen'][T])]] = CC
            CC += 1
    for k, v in icews_data['valid']['ent2id'].items():
        if k in icews_data['train']['ent2id']:
            icews_data['valid']['ent_map_list'].append(icews_data['train']['ent2id'][k])
        else:
            icews_data['valid']['ent_map_list'].append(-1)

    for k, v in icews_data['valid']['rel2id'].items():
        if k in icews_data['train']['rel2id']:
            icews_data['valid']['rel_map_list'].append(icews_data['train']['rel2id'][k])
        else:
            icews_data['valid']['rel_map_list'].append(-1)

    for k, v in icews_data['valid']['time2id'].items():
        if k in icews_data['train']['time2id']:
            icews_data['valid']['time_map_list'].append(icews_data['train']['time2id'][k])
        else:
            icews_data['valid']['time_map_list'].append(-1)

    src_list, dis_list = KG_valid.edges()
    rel_list = KG_valid.edata['relation_zhen']
    time_list = KG_valid.edata['time_zhen']
    for i in range(KG_valid.num_edges()):
        icews_data['valid']['support'].append((int(src_list[i]),
                                                   icews_data['valid']['rel2id'][all_id2_relation[int(rel_list[i])]],
                                                   int(dis_list[i]),
                                                   icews_data['valid']['time2id'][all_id2_time[int(time_list[i])]]))
        if not (icews_data['valid']['ent_map_list'][int(src_list[i])] != -1 and \
                   icews_data['valid']['ent_map_list'][int(dis_list[i])] != -1 \
                   and icews_data['valid']['rel_map_list'][
                       icews_data['valid']['rel2id'][all_id2_relation[int(rel_list[i])]]] != -1 and \
                   icews_data['valid']['time_map_list'][icews_data['valid']['time2id'][  all_id2_time[int(time_list[i]) ]  ]] != -1):

            icews_data['valid']['query'].append((int(src_list[i]),
                                                 icews_data['valid']['rel2id'][all_id2_relation[int(rel_list[i])]],
                                                 int(dis_list[i]),
                                                 icews_data['valid']['time2id'][all_id2_time[int(time_list[i])]]))
    print(1)
    with open("./icews14.pickle", "wb") as fp:  # Pickling
        pickle.dump(icews_data, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return icews_data

build_icews_data()