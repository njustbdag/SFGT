from torch_geometric.datasets import Reddit
from Utils.data import *
from scipy.io import loadmat

def normalize(mx):
    return mx


yelp = loadmat('../data/dataset/YelpChi.mat')
yelp_homo = yelp['homo']
yelp_rur = yelp['net_rur']
yelp_rsr = yelp['net_rsr']
yelp_rtr = yelp['net_rtr']
graph_Yelp = Graph()

rur_adj = yelp_rur + sp.eye(yelp_rur.shape[0])
rur_adj = rur_adj.nonzero()
rsr_adj = yelp_rsr + sp.eye(yelp_rsr.shape[0])
rsr_adj = rsr_adj.nonzero()
rtr_adj = yelp_rtr + sp.eye(yelp_rtr.shape[0])
rtr_adj = rtr_adj.nonzero()
homo_adj = yelp_homo + sp.eye(yelp_homo.shape[0])
homo_adg = homo_adj.nonzero()
el = defaultdict(  #target_id
                    lambda: defaultdict( #source_id(
                        lambda: int # time
                    ))

for index,node in enumerate(rur_adj[0]):
    source = {'id': node, 'type': 'def'}
    target = {'id': rur_adj[1][index], 'type': 'def'}
    if node != rur_adj[1][index]:
        graph_Yelp.add_edge(source,target,1,'rur',False)

for index,node in enumerate(rsr_adj[0]):
    source = {'id': node, 'type': 'def'}
    target = {'id': rsr_adj[1][index], 'type': 'def'}
    if node != rsr_adj[1][index]:
        graph_Yelp.add_edge(source,target,1,'rsr',False)

for index,node in enumerate(rtr_adj[0]):
    source = {'id': node, 'type': 'def'}
    target = {'id': rtr_adj[1][index], 'type': 'def'}
    if node != rtr_adj[1][index]:
        graph_Yelp.add_edge(source, target, 1, 'rtr', False)

for index,node in enumerate(homo_adg[0]):
    el[node][homo_adg[1][index]] = 1
    if node != homo_adg[1][index]:
        el[homo_adg[1][index]][node] = 1



target_type = 'def'
# graph_Yelp.edge_list['def']['def']['def'] = el
n = list(el.keys())
degree = np.zeros(np.max(n)+1)
for i in n:
    degree[i] = len(el[i])
x = yelp['feat']
x = normalize(x)
# the def denote the node type,but all the nodes in yelp dataset are the human,so they are denoted as def
graph_Yelp.node_feature['def'] = pd.DataFrame({'emb': list(x)})

# print(graph_Yelp.node_feature['def'])
idx = np.arange(len(graph_Yelp.node_feature[target_type]))
np.random.seed(43)
np.random.shuffle(idx)

graph_Yelp.train_target_nodes = idx[int(len(idx) * 0.00) : int(len(idx) * 0.60)]
graph_Yelp.valid_target_nodes = idx[int(len(idx) * 0.6) : int(len(idx) * 0.8)]
graph_Yelp.test_target_nodes  = idx[int(len(idx) * 0.8) : ]

graph_Yelp.y = yelp['label'].flatten()
dill.dump(graph_Yelp, open('../data/yelpchi/graph_Yelp_heter.pk', 'wb'))
print("yelp heter preprocess finish!")
