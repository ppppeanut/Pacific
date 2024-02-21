import random

import torch
import torch.nn as nn
from torch.nn import MultiheadAttention, init

from modules import LayerNorm, SlidingWindowMultiheadAttention, FeedForward


class PacificModel(nn.Module):
    def __init__(self, args):
        super(PacificModel, self).__init__()
        self.args = args
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.delta = nn.Parameter(torch.randn(args.hidden_size))
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.window_size = args.window_size
        self.last_N = args.last_N
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.gl_feed_forward = FeedForward(
            args.hidden_size,
            2 * args.hidden_size,
            args.hidden_dropout_prob,
            args.hidden_act,
            layer_norm_eps=1e-12,
        )
        self.ll_feed_forward = FeedForward(
            args.hidden_size,
            2 * args.hidden_size,
            args.hidden_dropout_prob,
            args.hidden_act,
            layer_norm_eps=1e-12,
        )
        self.mh_att = MultiheadAttention(args.hidden_size, args.num_heads, batch_first=True,
                                         dropout=args.attention_probs_dropout_prob)
        self.sw_mh_att = SlidingWindowMultiheadAttention(args.hidden_size, args.num_heads, self.window_size,
                                                         args.attention_probs_dropout_prob)

        self.w1 = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.w2 = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.w3 = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.w0 = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.b_a = nn.Parameter(torch.zeros(args.hidden_size), requires_grad=True)
        self.w4 = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.w5 = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.w6 = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.w7 = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.b_b = nn.Parameter(torch.zeros(args.hidden_size), requires_grad=True)

        self.sigmoid = nn.Sigmoid()

        self.apply(self.init_weights)

    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    def CPN_gl(self, context, aspect, output, args):
        r"""This is a function that count the attention weights

        Args:
            context(torch.FloatTensor): Item list embedding matrix, shape of [batch_size, time_steps, emb]
            aspect(torch.FloatTensor): The embedding matrix of the last click item, shape of [batch_size, emb]
            output(torch.FloatTensor): The average of the context, shape of [batch_size, emb]

        Returns:
            torch.Tensor:attention weights, shape of [batch_size, time_steps]
        """
        timesteps = context.size(1)
        aspect_3dim = aspect.repeat(1, timesteps).view(
            -1, timesteps, args.hidden_size
        )
        output_3dim = output.repeat(1, timesteps).view(
            -1, timesteps, args.hidden_size
        )
        res_ctx = self.w1(context)
        res_asp = self.w2(aspect_3dim)
        res_output = self.w3(output_3dim)
        res_sum = res_ctx + res_asp + res_output + self.b_a
        res_act = self.w0(self.sigmoid(res_sum))
        alpha = res_act
        return alpha

    def CPN_ll(self, context, aspect, output, args):
        r"""This is a function that count the attention weights

        Args:
            context(torch.FloatTensor): Item list embedding matrix, shape of [batch_size, time_steps, emb]
            aspect(torch.FloatTensor): The embedding matrix of the last click item, shape of [batch_size, emb]
            output(torch.FloatTensor): The average of the context, shape of [batch_size, emb]

        Returns:
            torch.Tensor:attention weights, shape of [batch_size, time_steps]
        """
        timesteps = context.size(1)
        aspect_3dim = aspect.repeat(1, timesteps).view(
            -1, timesteps, args.hidden_size
        )
        output_3dim = output.repeat(1, timesteps).view(
            -1, timesteps, args.hidden_size
        )
        res_ctx = self.w4(context)
        res_asp = self.w5(aspect_3dim)
        res_output = self.w6(output_3dim)
        res_sum = res_ctx + res_asp + res_output + self.b_b
        res_act = self.w7(self.sigmoid(res_sum))
        beta = res_act
        return beta

    def sim_loss(self, output1, output2):
        scores = torch.bmm(output1, output2.transpose(1, 2))  # [2B 2B]
        output1_l2 = torch.norm(output1, dim=-1).unsqueeze(-1)  # [2B]
        output2_l2 = torch.norm(output2, dim=-1).unsqueeze(-1)  # [2B]
        scores_l2 = torch.bmm(output1_l2, output2_l2.transpose(1, 2))  # [2B 2B]
        sim = scores / scores_l2  # [2B 2B]
        # sim_loss = torch.mean(torch.abs(1 - sim))
        sim_loss = torch.mean(torch.exp(sim))

        return sim_loss

    def CFSampler(self, input_ids):
        batch_size, seq_len = input_ids.size()
        sequence_emb = self.add_position_embedding(input_ids)

        # 生成随机索引
        random_indices = torch.randint(0, seq_len, size=(batch_size,)).to(input_ids.device)

        # 在item_e中的随机一项上加上item_distance
        sampled_tensor = sequence_emb.clone()  # 克隆item_e，以免修改原始张量
        sampled_tensor[torch.arange(batch_size), random_indices] += self.delta  # 加上item_distance的对应项

        return sampled_tensor

    def randomSampler(self, input_ids):
        item_size = self.args.item_size
        batch_size, seq_len = input_ids.size()

        item_set = list(range(1, item_size))  # 不-1是对的

        random_indices = torch.randint(0, seq_len, size=(batch_size,)).to(input_ids.device)

        replacement_items = torch.tensor(random.sample(item_set, batch_size)).to(input_ids.device)

        sampled_tensor = input_ids.scatter(1, random_indices.view(-1, 1), replacement_items.view(-1, 1))
        return sampled_tensor

    def forward(self, input_ids):

        random_sequence_sampled = self.randomSampler(input_ids)
        random_sequence_sampled_1 = self.randomSampler(input_ids)
        CF_sequence_sampled_emb = self.CFSampler(input_ids)
        input_ids = torch.cat([input_ids, random_sequence_sampled], dim=0)
        input_ids = torch.cat([input_ids, random_sequence_sampled_1], dim=0)

        sequence_emb = self.add_position_embedding(input_ids)
        sequence_emb = torch.cat([sequence_emb, CF_sequence_sampled_emb], dim=0)

        m_gl, _ = self.mh_att(sequence_emb, sequence_emb, sequence_emb)
        m_ll, _ = self.sw_mh_att(sequence_emb, sequence_emb, sequence_emb)

        last_inputs = sequence_emb[:, -1, :]
        org_memory = sequence_emb
        ms_gl = torch.div(torch.sum(m_gl, dim=1), input_ids.size(1))
        ms_ll = torch.div(torch.sum(m_ll, dim=1), input_ids.size(1))



        ce_gl = self.CPN_gl(org_memory, last_inputs, ms_gl, args=self.args)
        ce_ll = self.CPN_ll(org_memory, last_inputs, ms_ll, args=self.args)

        # 残差连接
        f_gl = self.LayerNorm(sequence_emb + m_gl)
        f_gl = self.gl_feed_forward(f_gl)
        f_gl = self.LayerNorm(sequence_emb + f_gl)
        f_ll = self.LayerNorm(sequence_emb + m_ll)
        f_ll = self.ll_feed_forward(f_ll)
        f_ll = self.LayerNorm(sequence_emb + f_ll)

        sim_loss = self.sim_loss(f_gl, f_ll)

        last_inputs = sequence_emb[:, -self.last_N:, :].sum(1)
        sequence_output = ce_gl * f_gl + ce_ll * f_ll + \
                          last_inputs.expand(f_ll.shape[1], last_inputs.shape[0],
                                             last_inputs.shape[1]).transpose(0, 1)

        return sequence_output, sim_loss

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        # elif isinstance(module, nn.Parameter):
        #     init.normal_(module, mean=0.0, std=0.5)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
