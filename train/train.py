import sys

import numpy as np
import os

import torch.cuda
from numba import cuda
from Utils.data import *
from Utils.model import *
from warnings import filterwarnings
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity

from torchviz import make_dot,make_dot_from_trace

filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser(description='classification task')

'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default='../data/yelpchi',
                    help='The address of metadata.')
parser.add_argument('--dataset', type=str, default='graph_YelpChi_heter.pk',
                    help='The address of graph.')
parser.add_argument('--model_dir', type=str, default='../Models',
                    help='The address for storing the models and optimization results.')
parser.add_argument('--task_name', type=str, default='yelpchi',
                    help='The name of task.')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')
parser.add_argument('--sample_depth', type=int, default=6,
                    help='How many numbers to sample the graph')
parser.add_argument('--sample_width', type=int, default=128,
                    help='How many nodes to be sampled per layer per type')
parser.add_argument('--moments', type=int, default=5,
                    help='how many orders of moments')

'''
   Model arguments 
'''
parser.add_argument('--conv_name', type=str, default='mmgt',
                    choices=['gcn', 'gat', 'sage','mmgt'],
                    help='The name of GNN. By default is Multi-order Graph Transformer (mmgt)')
parser.add_argument('--n_hid', type=int, default=400,
                    help='Number of hidden dimension,must be the integral multiple of the multi-heads')
parser.add_argument('--dis_func', type=str, default="cos",
                    help='Distance Function')
parser.add_argument('--sample_threshold', type=float, default=0.7,
                    help='proportation of sampling edges')
parser.add_argument('--or_threshold', type=int, default=1000,
                    help='the epoth begin to sample edges')
parser.add_argument('--n_heads', type=int, default=8,
                    help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=2,
                    help='Number of GNN layers')
parser.add_argument('--prev_norm', help='Whether to add layer-norm on the previous layers', action='store_true')
parser.add_argument('--last_norm', help='Whether to add layer-norm on the last layers', action='store_true')
parser.add_argument('--dropout', type=int, default=0.2,
                    help='Dropout ratio')

'''
    Optimization arguments
'''
parser.add_argument('--optimizer', type=str, default='adamw',
                    choices=['adamw', 'adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--scheduler', type=str, default='cosine',
                    help='Name of learning rate scheduler.', choices=['cycle', 'cosine'])
parser.add_argument('--n_epoch', type=int, default=100,
                    help='Number of epoch to run')
parser.add_argument('--n_batch', type=int, default=6,
                    help='Number of batch (sampled graphs) for each epoch')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of output nodes for training')
parser.add_argument('--clip', type=int, default=0.5,
                    help='Gradient Norm Clipping')

args = parser.parse_args()
args_print(args)


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

graph = renamed_load(open(os.path.join(args.data_dir, args.dataset), 'rb'))

target_type = 'def'
train_target_nodes = graph.train_target_nodes
valid_target_nodes = graph.valid_target_nodes
test_target_nodes = graph.test_target_nodes
''''''
threshold = args.sample_threshold
previous_dis = 1.0e99
types = graph.get_types()
criterion = nn.NLLLoss()

filename = f'../results/dataAnalysis.csv'

train_pos, train_neg = pos_neg_split(graph.train_target_nodes, graph.y[graph.train_target_nodes])

def cal_distance(x, y):
    if args.dis_func == "cos":
        return 1 - cosine_similarity(x, y)
    elif args.dis_func == "L1":
        return np.linalg.norm(x - y, ord=1)
    elif args.dis_func == "L2":
        return np.linalg.norm(x - y)


def node_classification_sample(seed, nodes, time_range, randomSample = True):
    np.random.seed(seed)
    if randomSample:
        samp_nodes = np.random.choice(nodes, args.batch_size, replace=False)
    else:
        samp_nodes = nodes[0:int(len(nodes) * 0.5)]
    feature, times, edge_list, _, texts = sample_subgraph(graph, time_range, \
                                                          inp={target_type: np.concatenate(
                                                              [samp_nodes, np.ones(args.batch_size)]).reshape(2,
                                                                                                              -1).transpose()}, \
                                                          sampled_depth=args.sample_depth,
                                                          sampled_number=args.sample_width,
                                                          feature_extractor=feature_reddit)

    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = \
        to_torch(feature, times, edge_list, graph)

    x_ids = np.arange(args.batch_size)
    return (node_feature, node_type, edge_time, edge_index, edge_type, x_ids, graph.y[samp_nodes])

stats = []
res = []
best_val = 0.0
train_step = 0
patience = 0
st = time.time()
'''
    Initialize GNN (model is specified by conv_name) and Classifier
'''
gnn = GNN(out_features=int(graph.y.max().item()) + 1 ,conv_name=args.conv_name, in_dim=len(graph.node_feature[target_type]['emb'].values[0]), n_hid=args.n_hid, \
          n_heads=args.n_heads, n_layers=args.n_layers, moments = args.moments, dropout=args.dropout, num_types=len(types), \
          num_relations=len(graph.get_meta_graph()) + 1, prev_norm=args.prev_norm, last_norm=args.last_norm,
          use_RTE=False)

classifier = Classifier(args.n_hid, int(graph.y.max().item()) + 1)

model = nn.Sequential(gnn, classifier).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

if args.scheduler == 'cycle':
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, pct_start=0.02, anneal_strategy='linear',
                                                    final_div_factor=100, \
                                                    max_lr=args.max_lr, total_steps=args.n_batch * args.n_epoch + 1)
elif args.scheduler == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500, eta_min=1e-6)

'''the fixed nodes for computing similarities of each epoch'''
sampled_idx_Simi = undersample(train_pos, train_neg, scale=1)
rd.shuffle(sampled_idx_Simi)
Simi_nodes = sampled_idx_Simi
SimiNodes_Sample = node_classification_sample(randint(), Simi_nodes, {1: True}, False)

for epoch in np.arange(args.n_epoch) + 1:

    train_data = []
    valid_data = []
    for batch_id in np.arange(args.n_batch):
        '''under sample'''
        sampled_idx_train = undersample(train_pos, train_neg, scale=1)
        rd.shuffle(sampled_idx_train)
        target_nodes = sampled_idx_train
        train_data.append(node_classification_sample(randint(), target_nodes, {1: True}))
        valid_data.append(node_classification_sample(randint(), valid_target_nodes, {1: True}))
        
    # valid_data = node_classification_sample(randint(), valid_target_nodes, {1: True})

    et = time.time()
    
    
    print('Data Preparation: %.1fs' % (et - st))

    '''
        Train
    '''
    model.train()
    train_losses = []
    ''''''
    if epoch >= args.or_threshold and args.or_threshold > 0:
        new_train_data = []
        for index, subgraph in enumerate(train_data):
          
            subgraph_node_features, subgraph_node_type, subgraph_edge_time, subgraph_edge_index, \
            subgraph_edge_type, subgraph_x_ids, subgraph_ylabel = subgraph
            
           
            '''H(L) of the Model in epoch-1'''
            subgraph_node_rep, subgraph_label_scores = gnn.forward(subgraph_node_features.to(device),
                                                                   subgraph_node_type.to(device), \
                                                                   subgraph_edge_time.to(device),
                                                                   subgraph_edge_index.to(device),
                                                                   subgraph_edge_type.to(device))
                                                                   
            node_res = classifier.forward(subgraph_node_rep)
            
            Dis = np.zeros((subgraph_node_rep.shape[0], subgraph_node_rep.shape[0]))
            adj_list = defaultdict(set)
            target_adj_edges = defaultdict(set)
            subgraph_edge_index = subgraph_edge_index.numpy()
            #"there"
            node_res = node_res.detach().cpu().numpy()
            #very slow")
            for id, i in enumerate(subgraph_edge_index[0]):
                j = subgraph_edge_index[1][id]
                x = node_res[i].reshape(1, -1)
                y = node_res[j].reshape(1, -1)
                t = cal_distance(x, y)
                Dis[i][j] = t
                if i < subgraph_x_ids.size:
                    '''edge_id,source_node,distance'''

                    temp = (tuple((id, j, Dis[i][j])))
                    target_adj_edges[i].add(tuple((id, j, Dis[i][j])))
            deleted_edges = []
            deleted_indexs = set()
            for id in range(subgraph_x_ids.size):
                neighbors = list(target_adj_edges[id])
                ''''''
                deleted_neighbors = sorted(neighbors, key=lambda x: list(x)[2])
                if (len(deleted_neighbors) > 1):
                    deleted_neighbors = deleted_neighbors[
                                        max(1, int(threshold * len(deleted_neighbors))):len(deleted_neighbors)]
                else:
                    deleted_neighbors = []
                deleted_edges.append(deleted_neighbors)
                for del_neighbor in deleted_neighbors:
                    deleted_indexs.add(del_neighbor[0])
            '''delete edges from edge_index,edge_time,edge_type'''
            np_index_ = subgraph_edge_index
            np_time_ = subgraph_edge_time.numpy()
            np_type_ = subgraph_edge_type.numpy()
            np_index = np.empty(shape=(2, 0), dtype=np.int64)
            np_time = np.empty(shape=(1, 0), dtype=np.int64)
            np_type = np.empty(shape=(1, 0), dtype=np.int64)
            counter = 0
            # d_i = list(deleted_indexs)
            for idx in range(len(np_time_)):
                if idx not in deleted_indexs:
                    np_index = np.column_stack((np_index, np_index_[:, idx]))
                    np_time = np.append(np_time, np_time_[idx])
                    np_type = np.append(np_type, np_type_[idx])
            new_tuple = (subgraph_node_features, subgraph_node_type, torch.tensor(np_time), torch.tensor(np_index), \
                         torch.tensor(np_type), subgraph_x_ids, subgraph_ylabel)
            new_train_data.append(new_tuple)

        train_data = new_train_data



    for node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel in train_data:
        '''H(L) of the Model in epoch'''
        node_feature = node_feature.to(device)
        node_type = node_type.to(device)
        edge_time = edge_time.to(device)
        edge_index = edge_index.to(device)
        edge_type = edge_type.to(device)

        node_rep, label_scores = gnn.forward(node_feature, node_type, edge_time, edge_index, edge_type)


        # graph = make_dot(node_rep,params=dict(model.named_parameters()),)
        # graph.render(filename='mmgt', view=False,format='png')



        '''get the gnn_scores by a classifier'''
        res = classifier.forward(node_rep[x_ids])

        ylabel = ylabel.astype(int)
        gnn_loss = criterion(res, torch.tensor(ylabel).to(device))
        label_loss = criterion(label_scores[x_ids], torch.tensor(ylabel).to(device))
        loss = gnn_loss

        optimizer.zero_grad()
        torch.cuda.empty_cache()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        train_losses = train_losses + [loss.cpu().detach().tolist()]
        train_step = train_step + 1
        scheduler.step(train_step)

        del node_feature, node_type, edge_time, edge_index, edge_type
        del res, loss



    '''update the sample_threshold according the similarity of sample nodes by the h(L) of the current epoch'''
    if epoch >= args.or_threshold and args.or_threshold > 0:
        simi_node_features, simi_node_type, simi_edge_time, simi_edge_index, \
        simi_edge_type, simi_x_ids, simi_ylabel = SimiNodes_Sample
        simi_ylabel = simi_ylabel.astype(int)
        simi_edge_time.data = (torch.ones(simi_edge_time.shape, dtype=torch.int64) * 120)
        simi_node_rep, simi_label_scores = gnn.forward(simi_node_features.to(device),
                                                               simi_node_type.to(device),
                                                               simi_edge_time.to(device),
                                                               simi_edge_index.to(device),
                                                               simi_edge_type.to(device))
        simi_node_res = classifier.forward(simi_node_rep).detach().cpu().numpy()
        same_class_dis = 0.0
        diff_class_dis = 0.0
        for id, i in enumerate(simi_edge_index[0]):
            j = simi_edge_index[1][id]
            x = simi_node_res[i].reshape(1, -1)
            y = simi_node_res[j].reshape(1, -1)
            t = cal_distance(x, y).item()
            if(simi_ylabel[i] == simi_ylabel[j]):
                same_class_dis += t
            else:
                diff_class_dis += t
        totdal_dis = same_class_dis - diff_class_dis
        threshold = threshold + ((previous_dis-totdal_dis)/abs(previous_dis)) * 0.01 
        previous_dis = totdal_dis
        if threshold > 1:
            threshold = 1
        elif threshold < 0:
            threshold = 0
        print(f"threshold=={threshold}")

    '''
        Valid

    '''
    model.eval()
    with torch.no_grad():
        avg_acc = 0.0
        avg_rec = 0.0
        avg_f1 = 0.0
        avg_auc = 0.0
        avg_loss = 0.0
        for node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel in valid_data:

            # node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = valid_data
            node_rep, label_scores = gnn.forward(node_feature.to(device), node_type.to(device), \
                                                 edge_time.to(device), edge_index.to(device), edge_type.to(device))

            res = classifier.forward(node_rep[x_ids])
            ylabel = ylabel.astype(int)
            loss = criterion(res, (torch.tensor(ylabel)).to(device))

            '''
                Calculate Valid F1. Update the best model based on highest F1 score.
            '''
            acc = accuracy_score(ylabel.tolist(), res.argmax(dim=1).cpu().tolist())
            rec = recall_score(ylabel.tolist(), res.argmax(dim=1).cpu().tolist())
            f1 = (2.0 * acc * rec) / (acc + rec)
            res = F.softmax(res, dim=-1)[:, 1].unsqueeze(1).cpu().numpy()
            auc_score = roc_auc_score(ylabel.tolist(), res)

            avg_acc += acc
            avg_rec += rec
            avg_f1 += f1
            avg_auc += auc_score
            avg_loss += loss

            del res, loss

        avg_acc = avg_acc / args.n_batch
        avg_rec = avg_rec / args.n_batch
        avg_f1 = avg_f1 / args.n_batch
        avg_auc = avg_auc / args.n_batch
        avg_loss = avg_loss / args.n_batch

        if avg_f1 > best_val:
            patience = 0
            best_val = avg_f1
            torch.save(model, os.path.join(args.model_dir, args.task_name + '_' + args.conv_name))
            print('UPDATE!!!')
        else:
            patience += 1

        st = time.time()
        print((
                  "Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  auc_score: %.4f recall: %.4f acc: %.4f  f1-score: %.4f") % \
              (epoch, (st - et), optimizer.param_groups[0]['lr'], np.average(train_losses), \
               avg_loss.cpu().detach().tolist(), avg_auc, avg_rec, avg_acc, avg_f1))
        stats += [[np.average(train_losses), avg_loss.cpu().detach().tolist()]]


    if patience == 1000: break
    del train_data, valid_data

    with open(f"{filename}", 'a+') as obj:
        obj.write(f"{args.conv_name}," + f"{args.dataset}," + f"{epoch}," + f"{args.n_layers}," +
                  f"{args.n_batch}," + f"{args.sample_width}," f"{args.sample_depth}," + f"{args.moments}," +
                  f"{args.n_heads}," + f"{args.n_epoch}," + f"{args.n_hid}," + f"{auc_score:.4f}," +
                  f"{rec:.4f}," + f"{acc:.4f}," + f"{f1:.4f}," + f"{np.average(train_losses):.4f}\n")
    if epoch == args.n_epoch:
        with open(f"{filename}", 'a+') as obj:
            obj.write(f"\n\n\n")




best_model = torch.load(os.path.join(args.model_dir, args.task_name + '_' + args.conv_name)).to(device)
best_model.eval()
gnn, classifier = best_model


with torch.no_grad():
    test_res = []
    test_acc = []
    fuc_acc = []
    test_recall = []
    test_auc = []
    t0 = 0.0

    for _ in range(10):
        node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = \
            node_classification_sample(randint(), test_target_nodes, {1: True})
        et = time.time()
        rep, scores = gnn.forward(node_feature.to(device), node_type.to(device), \
                                        edge_time.to(device), edge_index.to(device), edge_type.to(device))
        res = classifier.forward(rep[x_ids])
        st = time.time()
        acc = accuracy_score(ylabel.tolist(), res.argmax(dim=1).cpu().tolist())
        rec = recall_score(ylabel.tolist(), res.argmax(dim=1).cpu().tolist())
        test_f1 = (2.0 * acc * rec) / (acc + rec)
        res = F.softmax(res, dim=-1)[:, 1].unsqueeze(1).cpu().numpy()
        auc = roc_auc_score(ylabel.tolist(), res)
        test_acc += [acc]
        test_res += [test_f1]
        test_recall += [rec]
        test_auc += [auc]
        t0 = t0 + st - et


    print('running time: %.4f' % (t0))
    print('Best Test Acc: %.4f' % np.average(test_acc))
    print('Best Test Recall: %.4f' % np.average(test_recall))
    print('Best Test F1: %.4f' % np.average(test_res))
    print('Best Test Auc: %.4f' % np.average(test_auc))
    filename = f'../results/{str(args.dataset).split(".")[0]}.csv'
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(f"{args.conv_name}," + f"{args.dataset}," + f"{args.n_layers}," +
                        f"{args.dis_func}," + f"{args.or_threshold}," + f"{args.sample_threshold}," +
                        f"{args.n_batch}," + f"{args.sample_width}," f"{args.sample_depth}," + f"{args.moments}," + f"{t0}," +
                        f"{args.n_heads}," + f"{args.moments}," + f"{args.n_epoch}," + f"{args.n_hid}," + f"{np.average(test_acc):.4f}," +
                        f"{np.average(test_recall):.4f}," + f"{np.average(test_res):.4f}," +
                        f"{np.average(test_auc):.4f}\n")
print('finish')


