from torch_geometric.datasets import Reddit
from Utils.data import *
from scipy.io import loadmat

#def normalize(mx):
#    rowsum = np.array(mx.sum(1)) + 0.01
#    r_inv = np.power(rowsum,-1).flatten()
#    r_inv[np.isinf(r_inv)] = 0
#    r_mat_inv = sp.diags(r_inv)
#    return r_mat_inv.dot(mx)

def normalize(mx):
    return mx

amazon = loadmat('../data/dataset/Amazon.mat')
amazon_homo = amazon['homo']
amazon_upu = amazon['net_upu']
amazon_usu = amazon['net_usu']
amazon_uvu = amazon['net_uvu']
graph_Amazon = Graph()

upu_adj = amazon_upu + sp.eye(amazon_upu.shape[0])
upu_adj = upu_adj.nonzero()
usu_adj = amazon_usu + sp.eye(amazon_usu.shape[0])
usu_adj = usu_adj.nonzero()
uvu_adj = amazon_uvu + sp.eye(amazon_uvu.shape[0])
uvu_adj = uvu_adj.nonzero()
homo_adj = amazon_homo + sp.eye(amazon_homo.shape[0])
homo_adg = homo_adj.nonzero()
el = defaultdict(  #target_id
                    lambda: defaultdict( #source_id(
                        lambda: int # time
                    ))
for index,node in enumerate(upu_adj[0]):
    source = {'id': node, 'type': 'def'}
    target = {'id': upu_adj[1][index], 'type': 'def'}
    if node != upu_adj[1][index]:
        graph_Amazon.add_edge(source,target,1,'upu',False)

for index,node in enumerate(usu_adj[0]):
    source = {'id': node, 'type': 'def'}
    target = {'id': usu_adj[1][index], 'type': 'def'}
    if node != usu_adj[1][index]:
        graph_Amazon.add_edge(source,target,1,'usu',False)

for index,node in enumerate(uvu_adj[0]):
    source = {'id': node, 'type': 'def'}
    target = {'id': uvu_adj[1][index], 'type': 'def'}
    if node != uvu_adj[1][index]:
        graph_Amazon.add_edge(source, target, 1, 'uvu', False)

for index,node in enumerate(homo_adg[0]):
    el[node][homo_adg[1][index]] = 1
    if node != homo_adg[1][index]:
        el[homo_adg[1][index]][node] = 1

target_type = 'def'
# graph_Yelp.edge_list['def']['def']['def'] = el
n = list(el.keys())

'''
Add the logarithm of the degree of each node (post) to the attribute vector of that node in the last dimension. Then, transform it into a matrix 
x, where each row represents the attribute representation of a node.
'''
x = amazon['feat']
x = normalize(x)
# the def denote the node type,but all the nodes in yelp dataset are the human,so they are denoted as def
graph_Amazon.node_feature['def'] = pd.DataFrame({'emb': list(x)})

idx = np.arange(len(graph_Amazon.node_feature[target_type]))
np.random.seed(43)
np.random.shuffle(idx)

graph_Amazon.train_target_nodes = idx[int(len(idx) * 0.00) : int(len(idx) * 0.60)]
graph_Amazon.valid_target_nodes = idx[int(len(idx) * 0.6) : int(len(idx) * 0.8)]
graph_Amazon.test_target_nodes  = idx[int(len(idx) * 0.8) : ]

graph_Amazon.y = amazon['label'].flatten()
dill.dump(graph_Amazon, open('../data/amazon/graph_Amazon_heter.pk', 'wb'))
print("Amazon pretrain graph generated!")
