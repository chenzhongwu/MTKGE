import torch.nn as nn
import torch


class KGEModel(nn.Module):
    def __init__(self, args):
        super(KGEModel, self).__init__()
        self.args = args
        self.model_name = args.kge
        # self.nrelation = args.num_rel
        self.emb_dim = args.dim
        self.epsilon = 2.0

        self.gamma = torch.Tensor([args.gamma])
        self.xishu = torch.Tensor([args.xishu])

        self.embedding_range = torch.Tensor([(self.gamma.item() + self.epsilon) / args.dim])

        self.pi = 3.14159265358979323846

    def forward(self, sample, ent_emb, rel_emb, time_emb, mode='single'):
        self.entity_embedding = ent_emb
        self.relation_embedding = rel_emb
        self.time_embedding = time_emb

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1


            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)
            time = torch.index_select(
                self.time_embedding,
                dim=0,
                index=sample[:, 3]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part= sample
            if head_part != None:
                batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            if head_part == None:
                head = self.entity_embedding.unsqueeze(0)
            else:
                head = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=head_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)


            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)
            if tail_part == None:
                time = self.time_embedding.unsqueeze(0)
            else:
                time = torch.index_select(
                    self.time_embedding,
                    dim=0,
                    index=tail_part[:, 3]
                ).unsqueeze(1)
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            if tail_part != None:
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=head_part[:, 0]
                ).unsqueeze(1)


            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            time = torch.index_select(
                    self.time_embedding,
                    dim=0,
                    index=head_part[:, 3]
                ).unsqueeze(1)



            if tail_part == None:
                tail = self.entity_embedding.unsqueeze(0)
            else:
                tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=tail_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)

        elif mode == 'rel-batch':
            head_part, tail_part = sample
            if tail_part != None:
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 2]
            ).unsqueeze(1)

            if tail_part == None:
                relation = self.relation_embedding.unsqueeze(0)
            else:
                relation = torch.index_select(
                    self.relation_embedding,
                    dim=0,
                    index=tail_part[:, 3]
                ).view(batch_size, 1 , len(self.time_embedding[0]))

            if tail_part == None:
                time = self.time_embedding.unsqueeze(0)
            else:
                time = torch.index_select(
                    self.time_embedding,
                    dim=0,
                    index=tail_part.view(-1)
                ).view(batch_size,1 , len(self.time_embedding[0]))

        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'TTransE': self.TTransE,
            'TDistMult': self.TDistMult,
            'TComplEx': self.TComplEx,
            'TeRo': self.TeRo
        }

        if self.model_name in model_func:
            score, reg = model_func[self.model_name](head, relation, tail, time, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score, reg

    def TransE(self, head, relation, tail, time, mode):
        if mode == 'head-batch':

            score = head + (relation - tail)
        else:
            score = (head + relation) - tail
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score,None

    def TTransE(self, head, relation, tail, time, mode):
        if mode == 'head-batch':

            score = head + (relation + time - tail)
        else:
            score = (head + relation + time) - tail
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score, None


    def DistMult(self, head, relation, tail, time, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        reg = (
            torch.sqrt(head ** 2), torch.sqrt(relation ** 2), torch.sqrt(tail ** 2)
        )
        return score,  reg

    def TDistMult(self, head, relation, tail, time, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)* time
        else:
            score = (head * relation) * tail* time

        score = score.sum(dim=2)

        reg = (
                torch.sqrt(head**2),torch.sqrt(relation**2),torch.sqrt(tail**2),torch.sqrt(time**2)
        )
        return score, reg


    def ComplEx(self, head, relation, tail, time, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score,None


    def TComplEx(self, head, relation, tail, time, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        re_time, im_time = torch.chunk(time, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail * re_time + im_relation * im_tail * re_time
            im_score = re_relation * im_tail * im_time - im_relation * re_tail * im_time
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation * re_time - im_head * im_relation * re_time
            im_score = re_head * im_relation * im_time + im_head * re_relation * im_time
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score,None

    def RotatE(self, head, relation, tail, time, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        pi = 3.14159265358979323846
        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score,None

    def TeRo(self, head, relation, tail, time, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        re_time, im_time = torch.chunk(time, 2, dim=2)

        pi = 3.14159265358979323846
        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail * re_time + im_relation * im_tail * im_time
            im_score = re_relation * im_tail * im_time - im_relation * re_tail * re_time
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_time * re_relation - im_head * im_time * im_relation
            im_score = re_head * re_time * im_relation + im_head * im_time * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score,None
