import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
import math

# torch.autograd.set_detect_anomaly(True)
class MMGTConv(MessagePassing):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, moments, dropout=0.2, use_norm=True, use_RTE=True,
                 **kwargs):
        super(MMGTConv, self).__init__(aggr='add', **kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_types = num_types
        self.num_relations = num_relations
        self.total_rel = num_types * num_relations * num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.att = None
        self.K = moments
        self.edge_index  = None
        self.node_inp = None

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.res_linear = nn.Linear(out_dim * self.K, out_dim)
        self.query_linears = nn.ModuleList()
        self.key_linears = nn.ModuleList()

        for k in range(self.K):
            self.query_linears.append(nn.Linear(out_dim, out_dim))
            self.key_linears.append(nn.Linear(out_dim, out_dim))

        self.query_linear = nn.Linear(out_dim, out_dim)
        self.key_linear = nn.Linear(out_dim, out_dim)
        self.test_linear = nn.Linear(out_dim, out_dim)

        self.norms = nn.ModuleList()


        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.WMk = nn.Parameter(torch.Tensor(self.K, out_dim, out_dim))
        self.Wak = nn.Parameter(torch.Tensor(self.K, out_dim, out_dim))

        self.bias = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

        self.skip = nn.Parameter(torch.ones(num_types))
        self.drop = nn.Dropout(dropout)

        glorot(self.relation_att)
        glorot(self.relation_msg)

        glorot(self.WMk)
        glorot(self.Wak)

    def forward(self, node_inp, node_type, edge_index, edge_type, edge_time):
        self.edge_index = edge_index
        self.node_inp = node_inp
        return self.propagate(edge_index, node_inp=node_inp, node_type=node_type, \
                              edge_type=edge_type, edge_time=edge_time)

    def message(self, edge_index_i, node_inp_i, node_inp_j, node_type_i, node_type_j, edge_type, edge_time):
        '''
            j: source, i: target; <j, i>
        '''
        data_size = edge_index_i.size(0)
        '''
            Create Attention and Message tensor beforehand.
        '''
        res_att = torch.zeros(data_size, self.n_heads).to(node_inp_i.device)
        res_msg = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)

        for source_type in range(self.num_types):
            sb = (node_type_j == int(source_type))
            k_linear = self.k_linears[source_type]
            v_linear = self.v_linears[source_type]
            for target_type in range(self.num_types):
                tb = (node_type_i == int(target_type)) & sb
                q_linear = self.q_linears[target_type]
                for relation_type in range(self.num_relations):
                    '''
                        idx is all the edges with meta relation <source_type, relation_type, target_type>
                    '''
                    idx = (edge_type == int(relation_type)) & tb
                    if idx.sum() == 0:
                        continue
                    '''
                        Get the corresponding input node representations by idx.
                    '''
                    '''"target_node_vec" represents the embeddings of all nodes on the "i" end under the <source_type, relation_type, target_type> relationship.'''
                    target_node_vec = node_inp_i[idx]

                    source_node_vec = node_inp_j[idx]

                    '''
                        Step 1: Mutual Attention
                    '''
                    q_mat = q_linear(target_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = k_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = torch.bmm(k_mat.transpose(1, 0), self.relation_att[relation_type]).transpose(1, 0)
                    res_att[idx] = (q_mat * k_mat).sum(dim=-1) * self.relation_pri[relation_type] / self.sqrt_dk
                    '''
                        Step 2: Message Passing
                    '''
                    v_mat = v_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    res_msg[idx] = torch.bmm(v_mat.transpose(1, 0), self.relation_msg[relation_type]).transpose(1, 0)

        self.att = softmax(res_att, edge_index_i)

        # Expand the original matrix based on the value of K for calculating multi-order matrix information.
        expand_res_msg = torch.zeros((self.K,) + res_msg.size()).to(res_msg.device)
        for k in range(self.K):
            expand_res_msg[k] = torch.pow(res_msg, k + 1)

        res = expand_res_msg * self.att.view(-1, self.n_heads, 1)

        # res = res_msg * self.att.view(-1, self.n_heads, 1)
        del res_att, res_msg
        return res.view(self.K, -1, self.out_dim)


    def aggregate(self, inputs, index, ptr, dim_size):
        aggregate_msg = torch.zeros(self.K, dim_size, inputs.size(2)).to(inputs.device)
        attention_moment = torch.zeros(self.K, dim_size).to(inputs.device)
        res = torch.zeros(dim_size, self.out_dim).to(inputs.device)
        # res = torch.empty(dim_size, 0).to(inputs.device)

        #Eq.9
        for k in range(self.K):
            aggregate_msg[k] = torch.zeros_like(aggregate_msg[k])

            neighbor_indices = self.edge_index[1]
            messages = inputs[k][neighbor_indices]
            aggregate_msg[k].index_add_(0, index, messages)


            aggregate_message = aggregate_msg.clone()

            # t = torch.pow(torch.abs(aggregate_msg[k]) + 1e-18, 1 / (k + 1))



            aggregate_message[k] = torch.pow(torch.abs(aggregate_msg[k]) + 1e-18, 1 / (k + 1)) * torch.sign(aggregate_msg[k])
            '''
            for i in range(self.edge_index.size(1)):
                neighbor_index = self.edge_index[1][i]
                message = inputs[k][neighbor_index]
                aggregate_message[k][index[i]] = aggregate_message[k][index[i]] + message
            '''

        aggregate_message = torch.bmm(aggregate_message, self.WMk)


        att_output = torch.tensor([]).to(inputs.device)
        #Eq.10 & 11: attention & aggregate
        for k in range(self.K):
            front = torch.matmul(self.query_linear(self.node_inp), self.Wak[k]).unsqueeze(1)
            tail = self.key_linear(aggregate_message[k]).unsqueeze(2)
            attention_moment[k] = F.sigmoid(torch.bmm(front, tail).squeeze())
            t = attention_moment[k].clone()

            res = res + t.view(-1, 1) * aggregate_message[k]
            # res = torch.cat([res, t.view(-1, 1) * aggregate_message[k]], dim=1)

            att_output = torch.cat((att_output, t.view(-1, 1)), dim=1)

        # att_output = F.softmax(att_output)
        # for k in range(self.K):
        #     res = res + att_output[:, k].view(-1, 1) * aggregate_message[k]
        '''
        attention visualization
        import global_vars
        if global_vars.train_flag == 0 and global_vars.output_flag <= 5:
            if global_vars.current_layers == 2:
                filename = f'../results/attAnalysis.csv'
                att_output = att_output.T
                with open(f"{filename}", 'a+') as write_obj:
                    for i in range(len(att_output)):
                        i = torch.tensor(i, dtype=torch.long)
                        for j in range(40):
                            j = torch.tensor(j, dtype=torch.long)
                            write_obj.write(f"{att_output[i, j]},")
                        write_obj.write(f"\n")
                    write_obj.write(f"\n")
                global_vars.output_flag = global_vars.output_flag + 1
            else:
                global_vars.current_layers += 1
        '''

        return res
        # return self.res_linear(res)


    def update(self, aggr_out, node_inp, node_type):
        aggr_out = F.gelu(aggr_out)
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)
        for target_type in range(self.num_types):
            idx = (node_type == int(target_type))
            if idx.sum() == 0:
                continue
            trans_out = self.a_linears[target_type](aggr_out[idx])
            alpha = torch.sigmoid(self.skip[target_type])
            if self.use_norm:
                res[idx] = self.norms[target_type](trans_out * alpha + node_inp[idx] * (1 - alpha))
            else:
                res[idx] = trans_out * alpha + node_inp[idx] * (1 - alpha)
        return self.drop(res)

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)


    
class GeneralConv(nn.Module):
    def __init__(self, conv_name, in_hid, out_hid, num_types, num_relations, n_heads, moments, dropout, use_norm = True, use_RTE = True):
        super(GeneralConv, self).__init__()
        self.conv_name = conv_name
        if self.conv_name == 'gcn':
            self.base_conv = GCNConv(in_hid, out_hid)
        elif self.conv_name == 'gat':
            self.base_conv = GATConv(in_hid, out_hid // n_heads, heads=n_heads)
        elif self.conv_name == 'mmgt':
            self.base_conv = MMGTConv(in_hid, out_hid, num_types, num_relations, n_heads, moments, dropout, use_norm, use_RTE)
    def forward(self, meta_xs, node_type, edge_index, edge_type, edge_time):
        if self.conv_name == 'gcn':
            return self.base_conv(meta_xs, edge_index)
        elif self.conv_name == 'gat':
            return self.base_conv(meta_xs, edge_index)
        elif self.conv_name == 'mmgt':
            return self.base_conv(meta_xs, node_type, edge_index, edge_type, edge_time)
    
  
