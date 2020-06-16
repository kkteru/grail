import pdb
import os
import math
import random
import argparse
import numpy as np

from graph_utils import incidence_matrix, get_edge_count
from dgl_utils import _bfs_relational
from data_utils import process_files, save_to_file


def get_active_relations(adj_list):
    act_rels = []
    for r, adj in enumerate(adj_list):
        if len(adj.tocoo().row.tolist()) > 0:
            act_rels.append(r)
    return act_rels


def get_avg_degree(adj_list):
    adj_mat = incidence_matrix(adj_list)
    degree = []
    for node in range(adj_list[0].shape[0]):
        degree.append(np.sum(adj_mat[node, :]))
    return np.mean(degree)


def get_splits(adj_list, nodes, valid_rels=None, valid_ratio=0.1, test_ratio=0.1):
    '''
    Get train/valid/test splits of the sub-graph defined by the given set of nodes. The relations in this subbgraph are limited to be among the given valid_rels.
    '''

    # Extract the subgraph
    subgraph = [adj[nodes, :][:, nodes] for adj in adj_list]

    # Get the relations that are allowed to be sampled
    active_rels = get_active_relations(subgraph)
    common_rels = list(set(active_rels).intersection(set(valid_rels)))

    print('Average degree : ', get_avg_degree(subgraph))
    print('Nodes: ', len(nodes))
    print('Links: ', np.sum(get_edge_count(subgraph)))
    print('Active relations: ', len(common_rels))

    # get all the triplets satisfying the given constraints
    all_triplets = []
    for r in common_rels:
        # print(r, len(subgraph[r].tocoo().row))
        for (i, j) in zip(subgraph[r].tocoo().row, subgraph[r].tocoo().col):
            all_triplets.append([nodes[i], nodes[j], r])
    all_triplets = np.array(all_triplets)

    # delete the triplets which correspond to self connections
    ind = np.argwhere(all_triplets[:, 0] == all_triplets[:, 1])
    all_triplets = np.delete(all_triplets, ind, axis=0)
    print('Links after deleting self connections : %d' % len(all_triplets))

    # get the splits according to the given ratio
    np.random.shuffle(all_triplets)
    train_split = int(math.ceil(len(all_triplets) * (1 - valid_ratio - test_ratio)))
    valid_split = int(math.ceil(len(all_triplets) * (1 - test_ratio)))

    train_triplets = all_triplets[:train_split]
    valid_triplets = all_triplets[train_split: valid_split]
    test_triplets = all_triplets[valid_split:]

    return train_triplets, valid_triplets, test_triplets, common_rels


def get_subgraph(adj_list, hops, max_nodes_per_hop):
    '''
    Samples a subgraph around randomly chosen root nodes upto hops with a limit on the nodes selected per hop given by max_nodes_per_hop
    '''

    # collapse the list of adj mattricees to a single matrix
    A_incidence = incidence_matrix(adj_list)

    # chose a set of random root nodes
    idx = np.random.choice(range(len(A_incidence.tocoo().row)), size=params.n_roots, replace=False)
    roots = set([A_incidence.tocoo().row[id] for id in idx] + [A_incidence.tocoo().col[id] for id in idx])

    # get the neighbor nodes within a limit of hops
    bfs_generator = _bfs_relational(A_incidence, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(hops):
        lvls.append(next(bfs_generator))

    nodes = list(roots) + list(set().union(*lvls))

    return nodes


def mask_nodes(adj_list, nodes):
    '''
     mask a set of nodes from a given graph
    '''

    masked_adj_list = [adj.copy() for adj in adj_list]
    for node in nodes:
        for adj in masked_adj_list:
            adj.data[adj.indptr[node]:adj.indptr[node + 1]] = 0
            adj = adj.tocsr()
            adj.data[adj.indptr[node]:adj.indptr[node + 1]] = 0
            adj = adj.tocsc()
    for adj in masked_adj_list:
        adj.eliminate_zeros()
    return masked_adj_list


def main(params):

    adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(files)

    meta_train_nodes = get_subgraph(adj_list, params.hops, params.max_nodes_per_hop)  # list(range(750, 8500))  #

    masked_adj_list = mask_nodes(adj_list, meta_train_nodes)

    meta_test_nodes = get_subgraph(masked_adj_list, params.hops_test + 1, params.max_nodes_per_hop_test)  # list(range(0, 750))  #

    print('Common nodes among the two disjoint datasets (should ideally be zero): ', set(meta_train_nodes).intersection(set(meta_test_nodes)))
    tmp = [adj[meta_train_nodes, :][:, meta_train_nodes] for adj in masked_adj_list]
    print('Residual edges (should be zero) : ', np.sum(get_edge_count(tmp)))

    print("================")
    print("Train graph stats")
    print("================")
    train_triplets, valid_triplets, test_triplets, train_active_rels = get_splits(adj_list, meta_train_nodes, range(len(adj_list)))
    print("================")
    print("Meta-test graph stats")
    print("================")
    meta_train_triplets, meta_valid_triplets, meta_test_triplets, meta_active_rels = get_splits(adj_list, meta_test_nodes, train_active_rels)

    print("================")
    print('Extra rels (should be empty): ', set(meta_active_rels) - set(train_active_rels))

    # TODO: ABSTRACT THIS INTO A METHOD
    data_dir = os.path.join(params.main_dir, 'data/{}'.format(params.new_dataset))
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    save_to_file(data_dir, 'train.txt', train_triplets, id2entity, id2relation)
    save_to_file(data_dir, 'valid.txt', valid_triplets, id2entity, id2relation)
    save_to_file(data_dir, 'test.txt', test_triplets, id2entity, id2relation)

    meta_data_dir = os.path.join(params.main_dir, 'data/{}'.format(params.new_dataset + '_meta'))
    if not os.path.exists(meta_data_dir):
        os.makedirs(meta_data_dir)

    save_to_file(meta_data_dir, 'train.txt', meta_train_triplets, id2entity, id2relation)
    save_to_file(meta_data_dir, 'valid.txt', meta_valid_triplets, id2entity, id2relation)
    save_to_file(meta_data_dir, 'test.txt', meta_test_triplets, id2entity, id2relation)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Save adjacency matrtices and triplets')

    parser.add_argument("--dataset", "-d", type=str, default="FB15K237",
                        help="Dataset string")
    parser.add_argument("--new_dataset", "-nd", type=str, default="fb_v3",
                        help="Dataset string")
    parser.add_argument("--n_roots", "-n", type=int, default="1",
                        help="Number of roots to sample the neighborhood from")
    parser.add_argument("--hops", "-H", type=int, default="3",
                        help="Number of hops to sample the neighborhood")
    parser.add_argument("--max_nodes_per_hop", "-m", type=int, default="2500",
                        help="Number of nodes in the neighborhood")
    parser.add_argument("--hops_test", "-HT", type=int, default="3",
                        help="Number of hops to sample the neighborhood")
    parser.add_argument("--max_nodes_per_hop_test", "-mt", type=int, default="2500",
                        help="Number of nodes in the neighborhood")
    parser.add_argument("--seed", "-s", type=int, default="28",
                        help="Numpy random seed")

    params = parser.parse_args()

    np.random.seed(params.seed)
    random.seed(params.seed)

    params.main_dir = os.path.join(os.path.relpath(os.path.dirname(os.path.abspath(__file__))), '..')

    files = {
        'train': os.path.join(params.main_dir, 'data/{}/train.txt'.format(params.dataset)),
        'valid': os.path.join(params.main_dir, 'data/{}/valid.txt'.format(params.dataset)),
        'test': os.path.join(params.main_dir, 'data/{}/test.txt'.format(params.dataset))
    }

    main(params)
