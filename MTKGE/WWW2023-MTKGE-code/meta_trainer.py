from model import Model
from data import *
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import optim
import numpy as np
from trainer import Trainer
import os
from regularizers import N3

class MetaTrainer(Trainer):
    def __init__(self, args):
        super(MetaTrainer, self).__init__(args)
        # dataset
        self.train_subgraph_iter = OneShotIterator(DataLoader(TrainSubgraphDataset(args),
                                                   batch_size=self.args.train_bs,
                                                   shuffle=True,
                                                   collate_fn=TrainSubgraphDataset.collate_fn))

        # model
        self.model = Model(args).to(args.gpu)

        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.N3 = N3(args.reg)
        # args for controlling training
        self.num_step = args.num_step
        self.log_per_step = args.log_per_step
        self.check_per_step = args.check_per_step
        self.early_stop_patience = args.early_stop_patience

    def get_curr_state(self):
        state = {'model': self.model.state_dict()}
        return state

    def before_test_load(self):
        state = torch.load(os.path.join(self.state_path, self.name + '.best'), map_location=self.args.gpu)
        self.model.load_state_dict(state['model'])

    def get_loss(self, tri, neg_tail_ent, neg_head_ent, ent_emb, rel_emb, time_emb):

        neg_tail_score, reg_neg_tail = self.kge_model((tri, neg_tail_ent), ent_emb, rel_emb, time_emb, mode='tail-batch')
        neg_head_score, reg_neg_head = self.kge_model((tri, neg_head_ent), ent_emb, rel_emb, time_emb, mode='head-batch')

        neg_score = torch.cat((neg_tail_score, neg_head_score))
        a = neg_score
        neg_score = (F.softmax(neg_score * self.args.adv_temp, dim=1).detach()
                     * F.logsigmoid(-a)).sum(dim=1)


        pos_score, reg_pos = self.kge_model(tri, ent_emb, rel_emb, time_emb)

        pos_score = F.logsigmoid(pos_score).squeeze(dim=1)

        positive_sample_loss = - pos_score.mean()
        negative_sample_loss = - neg_score.mean()
        loss = (positive_sample_loss + negative_sample_loss) / 2
        return loss, reg_neg_tail, reg_neg_head, reg_pos

    def split_emb(self, emb, split_list):
        split_list = [np.sum(split_list[0: i], dtype=np.int) for i in range(len(split_list) + 1)]
        emb_split = [emb[split_list[i]: split_list[i + 1]] for i in range(len(split_list) - 1)]
        return emb_split

    def train_one_step(self):

        batch = next(self.train_subgraph_iter)
        batch_loss = 0
        batch_reg = 0
        # Batch a collection of DGLGraph s into one graph for more efficient graph computation. Each input graph becomes one disjoint component of the batched graph. The nodes and edges are relabeled to be disjoint segments:
        batch_pattern_g = dgl.batch([d[1] for d in batch]).to(self.args.gpu)


        for idx, d in enumerate(batch):

            d[0].edata['b_rel'] = d[0].edata['rel'] + torch.sum(batch_pattern_g.batch_num_nodes()[:idx]).cpu()

        batch_sup_g = dgl.batch([d[0] for d in batch]).to(self.args.gpu)


        batch_ent_emb, batch_rel_emb, batch_time_emb = self.model(batch_sup_g, batch_pattern_g)


        batch_ent_emb = self.split_emb(batch_ent_emb, batch_sup_g.batch_num_nodes().tolist())
        batch_rel_emb = self.split_emb(batch_rel_emb, batch_pattern_g.batch_num_nodes().tolist())
        l_reg=0
        for batch_i, data in enumerate(batch):

            a = data[2].cuda()
            b = data[3].cuda()
            c = data[4].cuda()
            [que_tri, que_neg_tail_ent, que_neg_head_ent] = [a, b, c]
            ent_emb = batch_ent_emb[batch_i]
            rel_emb = batch_rel_emb[batch_i]


            loss, reg_neg_tail, reg_neg_head, reg_pos = self.get_loss(que_tri, que_neg_tail_ent, que_neg_head_ent, ent_emb, rel_emb, batch_time_emb)
            if reg_neg_tail==None:
                l_reg += 0
            else:
                l_reg = self.N3.forward(reg_neg_tail, reg_neg_head, reg_pos)


            batch_loss += loss
            batch_reg  += l_reg


        batch_loss /= len(batch)
        batch_reg /= len(batch)
        batch_loss = batch_loss + batch_reg
        return batch_loss

    def get_eval_emb(self, eval_data):
        ent_emb, rel_emb, time_emb = self.model(eval_data.g, eval_data.pattern_g)

        return ent_emb, rel_emb, time_emb
