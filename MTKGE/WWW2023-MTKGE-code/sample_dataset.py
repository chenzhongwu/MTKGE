import os
import random
import numpy as np
from collections import defaultdict as ddict
from tqdm import tqdm
import pickle
import dgl
import torch
import copy
data_path = './data/icews05-15'
new_ratio = 0.2
def read_dict_openke(dict_path):


    element_dict = {}
    with open(dict_path, 'r') as f:
        f.readline()
        for line in f:
            element, id_ = line.strip().split('\t')
            element_dict[element] = int(id_)
    return element_dict

def read_data_openke(data_path):

    triples = []
    with open(data_path, 'r') as f:
        f.readline()
        for line in f:
            head, relation, tail, time = line.strip().split('\t')
            head = int(head)
            relation = int(relation)
            tail = int(tail)
            time = int(time)
            triples.append((head,relation, tail,time))
    return triples


entity_dict = read_dict_openke(os.path.join(data_path, 'entity2id.txt'))
relation_dict = read_dict_openke(os.path.join(data_path, 'relation2id.txt'))
time_dict = read_dict_openke(os.path.join(data_path, 'time2id.txt'))

entity_dict_inv = {v: k for k, v in entity_dict.items()}
relation_dict_inv = {v: k for k, v in relation_dict.items()}
time_dict_inv = {v: k for k, v in time_dict.items()}

train_triples = read_data_openke(os.path.join(data_path, 'icewsnum05train.txt'))
valid_triples = read_data_openke(os.path.join(data_path, 'icewsnum05valid.txt'))
test_triples = read_data_openke(os.path.join(data_path, 'icewsnum05test.txt'))


triples = train_triples + valid_triples + test_triples
triples = torch.tensor(triples)

# sample test triples
g_undir = dgl.graph((torch.cat([triples[:, 0], triples[:, 2]]),
                     torch.cat([triples[:, 2], triples[:, 0]])))

g = dgl.graph((triples[:, 0], triples[:, 2]))
g.edata['rel'] = triples[:, 1]
g.edata['time'] = triples[:, 3]
num_root_ent = 100
rw_len = 10


root_ent = np.random.choice(g_undir.num_nodes(), num_root_ent, replace=False)
random_ent = torch.unique(dgl.sampling.random_walk(g_undir, root_ent, length=rw_len)[0])
if -1 in random_ent:
    random_ent = random_ent[1:]

test_g = dgl.node_subgraph(g, random_ent)  # induce test triples from sampled entities

test_ent = test_g.ndata[dgl.NID]  # entity in test triples
test_rel = torch.unique(test_g.edata['rel'])  # relation in test triples
test_time = torch.unique(test_g.edata['time'])  # relation in test triples

test_new_ent = np.random.choice(test_ent, int(len(test_ent) * new_ratio), replace=False)  # entities that only appear in test triples
test_new_rel = np.random.choice(test_rel, int(len(test_rel) * new_ratio), replace=False)  # relations that only appear in test triples
test_new_time = np.random.choice(test_time, int(len(test_time) * new_ratio), replace=False)  # time that only appear in test triples


test_remain_edge = np.setdiff1d(np.arange(g.num_edges()), test_g.edata[dgl.EID])
test_remain_g = dgl.edge_subgraph(g, test_remain_edge)

test_remain_tri = torch.stack([test_remain_g.ndata[dgl.NID][test_remain_g.edges()[0]],
                               test_remain_g.edata['rel'],
                               test_remain_g.ndata[dgl.NID][test_remain_g.edges()[1]],
                               test_remain_g.edata['time']

                               ]).T.tolist()

test_remain_tri_delnew = []
for tri in tqdm(test_remain_tri):
    head, relation, tail, time = tri
    if head not in test_new_ent and tail not in test_new_ent and relation not in test_new_rel and time not in test_new_time:
        test_remain_tri_delnew.append(tri)



# sample valid triples
triples_new = torch.tensor(test_remain_tri_delnew)
g_undir = dgl.graph((torch.cat([triples_new[:, 0], triples_new[:, 2]]),
                     torch.cat([triples_new[:, 2], triples_new[:, 0]])))

g = dgl.graph((triples_new[:, 0], triples_new[:, 2]))
g.edata['rel'] = triples_new[:, 1]
g.edata['time'] = triples_new[:, 3]
root_ent = np.random.choice(g_undir.num_nodes(), num_root_ent, replace=False)
random_ent = torch.unique(dgl.sampling.random_walk(g_undir, root_ent, length=rw_len)[0])
if -1 in random_ent:
    random_ent = random_ent[1:]

valid_g = dgl.node_subgraph(g, random_ent)

valid_ent = valid_g.ndata[dgl.NID]
valid_rel = torch.unique(valid_g.edata['rel'])
valid_time = torch.unique(valid_g.edata['time'])

valid_new_ent = np.random.choice(valid_ent, int(len(valid_ent) * new_ratio), replace=False)
valid_new_rel = np.random.choice(valid_rel, int(len(valid_rel) * new_ratio), replace=False)
valid_new_time = np.random.choice(valid_time, int(len(valid_time) * new_ratio), replace=False)
valid_remain_edge = np.setdiff1d(np.arange(g.num_edges()), valid_g.edata[dgl.EID])
valid_remain_g = dgl.edge_subgraph(g, valid_remain_edge)

valid_remain_tri = torch.stack([valid_remain_g.ndata[dgl.NID][valid_remain_g.edges()[0]],
                                valid_remain_g.edata['rel'],
                                valid_remain_g.ndata[dgl.NID][valid_remain_g.edges()[1]],
                                valid_remain_g.edata['time']]).T.tolist()

valid_remain_tri_delnew = []
for tri in tqdm(valid_remain_tri):
    head, relation, tail, time = tri
    if head not in test_new_ent and tail not in test_new_ent and relation not in test_new_rel and time not in test_new_time:
        valid_remain_tri_delnew.append(tri)



# sample train triples
triples_new = torch.tensor(valid_remain_tri_delnew)
g_undir = dgl.graph((torch.cat([triples_new[:, 0], triples_new[:, 2]]),
                     torch.cat([triples_new[:, 2], triples_new[:, 0]])))

g = dgl.graph((triples_new[:, 0], triples_new[:, 2]))
g.edata['rel'] = triples_new[:, 1]
g.edata['time'] = triples_new[:, 3]
num_train_root_ent = 100
train_rw_len = 10

root_ent = np.random.choice(g_undir.num_nodes(), num_train_root_ent, replace=False)
random_ent = torch.unique(dgl.sampling.random_walk(g_undir, root_ent, length=train_rw_len)[0])
if -1 in random_ent:
    random_ent = random_ent[1:]

train_g = dgl.node_subgraph(g, random_ent)


# train_g
# re-index triples in train/valid/test
train_triples = torch.stack([train_g.ndata[dgl.NID][train_g.edges()[0]],
                               train_g.edata['rel'],
                               train_g.ndata[dgl.NID][train_g.edges()[1]],
                               train_g.edata['time']
                             ]
                            ).T.tolist()

test_triples = torch.stack([test_g.ndata[dgl.NID][test_g.edges()[0]],
                               test_g.edata['rel'],
                            test_g.ndata[dgl.NID][test_g.edges()[1]],
                               test_g.edata['time']

                            ]).T.tolist()

valid_triples = torch.stack([valid_g.ndata[dgl.NID][valid_g.edges()[0]],
                               valid_g.edata['rel'],
                              valid_g.ndata[dgl.NID][valid_g.edges()[1]],
                               valid_g.edata['time']

                             ]).T.tolist()


# re-index train triples
def reidx_train(triples):
    ent_reidx = dict()
    rel_reidx = dict()
    time_reidx = dict()

    entidx = 0
    relidx = 0
    timeidx = 0

    reidx_triples = []
    for tri in triples:
        head, relation, tail, time = tri
        if head not in ent_reidx.keys():
            ent_reidx[head] = entidx
            entidx += 1
        if tail not in ent_reidx.keys():
            ent_reidx[tail] = entidx
            entidx += 1
        if relation not in rel_reidx.keys():
            rel_reidx[relation] = relidx
            relidx += 1
        if time not in time_reidx.keys():
            time_reidx[time] = timeidx
            timeidx += 1

        reidx_triples.append((ent_reidx[head], rel_reidx[relation], ent_reidx[tail], time_reidx[time]))

    return reidx_triples, ent_reidx, rel_reidx, time_reidx

train_triples, train_ent_reidx, train_rel_reidx, train_time_reidx = reidx_train(train_triples)

train_ent2id = {entity_dict_inv[k]: v for k, v in train_ent_reidx.items()}
train_rel2id = {relation_dict_inv[k]: v for k, v in train_rel_reidx.items()}
train_time2id = {time_dict_inv[k]: v for k, v in train_time_reidx.items()}


# re-index valid/test triples

def reidx_eval(triples, train_ent_reidx, train_rel_reidx, train_time_reidx):
    ent_reidx = dict()
    rel_reidx = dict()
    time_reidx = dict()

    entidx = 0
    relidx = 0
    timeidx = 0

    ent_freq = ddict(int)
    rel_freq = ddict(int)
    time_freq = ddict(int)

    reidx_triples = []
    for tri in triples:
        head, relation, tail, time = tri
        if head not in ent_reidx.keys():
            ent_reidx[head] = entidx
            entidx += 1
        if tail not in ent_reidx.keys():
            ent_reidx[tail] = entidx
            entidx += 1
        if relation not in rel_reidx.keys():
            rel_reidx[relation] = relidx
            relidx += 1
        if time not in time_reidx.keys():
            time_reidx[time] = timeidx
            timeidx += 1

        ent_freq[ent_reidx[head]] += 1
        ent_freq[ent_reidx[tail]] += 1
        rel_freq[rel_reidx[relation]] += 1
        time_freq[time_reidx[time]] += 1

        reidx_triples.append((ent_reidx[head], rel_reidx[relation], ent_reidx[tail], time_reidx[time]))

    ent_reidx_inv = {v: k for k, v in ent_reidx.items()}
    rel_reidx_inv = {v: k for k, v in rel_reidx.items()}
    time_reidx_inv = {v: k for k, v in time_reidx.items()}

    ent_map_list = [train_ent_reidx[ent_reidx_inv[i]] if ent_reidx_inv[i] in train_ent_reidx.keys() else -1
                    for i in range(len(ent_reidx))]
    rel_map_list = [train_rel_reidx[rel_reidx_inv[i]] if rel_reidx_inv[i] in train_rel_reidx.keys() else -1
                    for i in range(len(rel_reidx))]
    time_map_list = [train_time_reidx[time_reidx_inv[i]] if time_reidx_inv[i] in train_time_reidx.keys() else -1
                    for i in range(len(time_reidx))]

    return reidx_triples, ent_freq, rel_freq,time_freq, ent_reidx, rel_reidx, time_reidx, ent_map_list, rel_map_list, time_map_list

valid_triples, valid_ent_freq, valid_rel_freq,valid_time_freq, valid_ent_reidx, valid_rel_reidx, valid_time_reidx,\
    valid_ent_map_list, valid_rel_map_list, valid_time_map_list = reidx_eval(valid_triples, train_ent_reidx, train_rel_reidx,train_time_reidx )

test_triples, test_ent_freq, test_rel_freq,test_time_freq, test_ent_reidx, test_rel_reidx, test_time_reidx,\
    test_ent_map_list, test_rel_map_list, test_time_map_list = reidx_eval(test_triples, train_ent_reidx, train_rel_reidx, train_time_reidx)


valid_ent2id = {entity_dict_inv[k]: v for k, v in valid_ent_reidx.items()}
valid_rel2id = {relation_dict_inv[k]: v for k, v in valid_rel_reidx.items()}
valid_time2id = {time_dict_inv[k]: v for k, v in valid_time_reidx.items()}

test_ent2id = {entity_dict_inv[k]: v for k, v in test_ent_reidx.items()}
test_rel2id = {relation_dict_inv[k]: v for k, v in test_rel_reidx.items()}
test_time2id = {time_dict_inv[k]: v for k, v in test_time_reidx.items()}

# split triples in valid/test into support and query

def split_triples(triples, ent_freq, rel_freq, time_freq, ent_map_list, rel_map_list, time_map_list):
    ent_freq = copy.deepcopy(ent_freq)
    rel_freq = copy.deepcopy(rel_freq)
    time_freq = copy.deepcopy(time_freq)

    support_triples = []
    query_triples = []
    query_uent = []
    query_urel = []
    query_uboth = []

    random.shuffle(triples)
    for idx, tri in enumerate(triples):
        head, relation, tail, time = tri
        test_flag = (ent_map_list[head] == -1 or ent_map_list[tail] == -1 or rel_map_list[relation] == -1)

        if (ent_freq[head] > 2 and ent_freq[tail] > 2 and rel_freq[relation] > 2 and time_freq[time] > 2) and test_flag:
            append_flag = False
            if ent_map_list[head] != -1 and ent_map_list[tail] != -1 and rel_map_list[relation] == -1:
                if len(query_urel) <= int(len(triples) * 0.1):
                    query_urel.append(tri)
                    append_flag = True
            elif (ent_map_list[head] == -1 or ent_map_list[tail] == -1) and rel_map_list[relation] != -1:
                if len(query_uent) <= int(len(triples) * 0.1):
                    query_uent.append(tri)
                    append_flag = True
            else:
                if len(query_uboth) <= int(len(triples) * 0.1):
                    query_uboth.append(tri)
                    append_flag = True

            if append_flag:
                ent_freq[head] -= 1
                ent_freq[tail] -= 1
                rel_freq[relation] -= 1
            else:
                support_triples.append(tri)
        else:
            support_triples.append(tri)

    return support_triples, query_uent, query_urel, query_uboth


valid_sup_tris, valid_que_uent, valid_que_urel, valid_que_uboth = split_triples(valid_triples,
                                                                                valid_ent_freq, valid_rel_freq, valid_time_freq,
                                                                                valid_ent_map_list, valid_rel_map_list, valid_time_map_list)
test_sup_tris, test_que_uent, test_que_urel, test_que_uboth = split_triples(test_triples,
                                                                            test_ent_freq, test_rel_freq, test_time_freq,
                                                                            test_ent_map_list, test_rel_map_list, test_time_map_list)
data_dict = {'train': {'triples': train_triples, 'ent2id': train_ent2id, 'rel2id': train_rel2id},
             'valid': {'support': valid_sup_tris, 'query': valid_que_uent + valid_que_urel + valid_que_uboth,
                       'ent_map_list': valid_ent_map_list, 'rel_map_list': valid_rel_map_list, 'time_map_list': valid_time_map_list,
                       'ent2id': valid_ent2id, 'rel2id': valid_rel2id},
             'test': {'support': test_sup_tris, 'query_uent': test_que_uent,
                      'query_urel': test_que_urel, 'query_uboth': test_que_uboth,
                      'ent_map_list': test_ent_map_list, 'rel_map_list': test_rel_map_list, 'time_map_list': test_time_map_list,
                      'ent2id': test_ent2id, 'rel2id': test_rel2id}}
pickle.dump(data_dict, open('data/icews05-15/icews05test_data.pkl', 'wb'))



# data statistic
load_data = pickle.load(open('data/icews05-15/icews05test_data.pkl', 'rb'))

valid_num_new_ent = np.sum(np.array(load_data['valid']['ent_map_list']) == -1)
valid_num_new_rel = np.sum(np.array(load_data['valid']['rel_map_list']) == -1)
test_num_new_ent = np.sum(np.array(load_data['test']['ent_map_list']) == -1)
test_num_new_rel = np.sum(np.array(load_data['test']['rel_map_list']) == -1)
print('train:')
print(f"num_ent: {len(load_data['train']['ent2id'])}")
print(f"num_rel: {len(load_data['train']['rel2id'])}")
print(f"num_tri: {len(load_data['train']['triples'])}")

print('valid:')
print(f"num_ent: {len(load_data['valid']['ent2id'])}(new: {valid_num_new_ent}, {valid_num_new_ent/len(load_data['valid']['ent2id']):.2})")
print(f"num_rel: {len(load_data['valid']['rel2id'])}(new: {valid_num_new_rel}, {valid_num_new_rel/len(load_data['valid']['rel2id']):.2})")
print(f"num_sup: {len(load_data['valid']['support'])}")
print(f"num_que: {len(load_data['valid']['query'])}")

print('test:')
print(f"num_ent: {len(load_data['test']['ent2id'])}(new: {test_num_new_ent}, {test_num_new_ent/len(load_data['test']['ent2id']):.2})")
print(f"num_rel: {len(load_data['test']['rel2id'])}(new: {test_num_new_rel}, {test_num_new_rel/len(load_data['test']['rel2id']):.2})")
print(f"num_sup: {len(load_data['test']['support'])}")
print(f"num_que_uent: {len(load_data['test']['query_uent'])}")
print(f"num_que_urel: {len(load_data['test']['query_urel'])}")
print(f"num_que_uboth: {len(load_data['test']['query_uboth'])}")






















