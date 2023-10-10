from .conv import *
torch.autograd.set_detect_anomaly(True)

class Classifier(nn.Module):
    def __init__(self, n_hid, n_out):
        super(Classifier, self).__init__()
        self.n_hid    = n_hid
        self.n_out    = n_out
        self.linear   = nn.Linear(n_hid,  n_out)
    def forward(self, x):
        tx = self.linear(x)
        return torch.log_softmax(tx.squeeze(), dim=-1)
    def __repr__(self):
        return '{}(n_hid={}, n_out={})'.format(
            self.__class__.__name__, self.n_hid, self.n_out)

class GNN(nn.Module):
    def __init__(self, out_features, in_dim, n_hid, num_types, num_relations, n_heads, n_layers, moments, dropout = 0.2, conv_name = 'hgt', prev_norm = False, last_norm = False, use_RTE = True):
        super(GNN, self).__init__()
        self.threshold = 0.5
        self.RL = True
        self.label_clf = nn.Linear(n_hid, out_features)
        self.gcs = nn.ModuleList()
        self.num_types = num_types
        self.GDS = []
        self.FES = []
        self.classifier = Classifier(n_hid, out_features)
        self.stop_flag = False
        self.in_dim    = in_dim
        self.n_hid     = n_hid
        self.adapt_ws  = nn.ModuleList()
        self.drop      = nn.Dropout(dropout)
        for t in range(num_types):
            self.adapt_ws.append(nn.Linear(in_dim, n_hid))
        for l in range(n_layers - 1):#the layes for gnn,all the hid_dim are 400
            self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, moments, dropout, use_norm = prev_norm, use_RTE = use_RTE))
        self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, moments, dropout, use_norm = last_norm, use_RTE = use_RTE))

    def forward(self, node_feature, node_type, edge_time, edge_index, edge_type):
        res = torch.zeros(node_feature.size(0), self.n_hid).to(node_feature.device)
        for t_id in range(self.num_types):
            '''get all the ids of this node type in this subgraph'''
            idx = (node_type == int(t_id))
            '''no thie type nodes,continue'''
            if idx.sum() == 0:
                continue
            '''transform these nodes features from 33 to 400 by nn.linear(33,400)'''
            res[idx] = torch.tanh(self.adapt_ws[t_id](node_feature[idx]))
        '''dropout = 0.2 debite every element in res has the probability of 0.2 to be zero'''
        meta_xs = self.drop(res)
        del res
        for gc in self.gcs:
            scores = self.classifier(meta_xs)
            meta_xs = gc(meta_xs, node_type, edge_index, edge_type, edge_time)
        return meta_xs, scores

