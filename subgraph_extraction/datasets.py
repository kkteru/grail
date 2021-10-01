from torch.utils.data import Dataset
import timeit
import os
import logging
import lmdb
import numpy as np
import json
import pickle
import dgl
import dgl.contrib.sampling
from utils.graph_utils import ssp_multigraph_to_dgl, incidence_matrix
from utils.data_utils import process_files, save_to_file, plot_rel_dist
from .graph_sampler import *
import pdb


def generate_subgraph_datasets(params, splits=['train', 'valid'], saved_relation2id=None, max_label_value=None):

    testing = 'test' in splits
    adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(params.file_paths, saved_relation2id)

    # plot_rel_dist(adj_list, os.path.join(params.main_dir, f'data/{params.dataset}/rel_dist.png'))

    data_path = os.path.join(params.main_dir, f'data/{params.dataset}/relation2id.json')
    if not os.path.isdir(data_path) and not testing:
        with open(data_path, 'w') as f:
            json.dump(relation2id, f)

    graphs = {}

    for split_name in splits:
        graphs[split_name] = {'triplets': triplets[split_name], 'max_size': params.max_links}

    # Sample train and valid/test links
    for split_name, split in graphs.items():
        logging.info(f"Sampling negative links for {split_name}")
        split['pos'], split['neg'] = sample_neg(adj_list, split['triplets'], params.num_neg_samples_per_link, max_size=split['max_size'], constrained_neg_prob=params.constrained_neg_prob)

    if testing:
        directory = os.path.join(params.main_dir, 'data/{}/'.format(params.dataset))
        save_to_file(directory, f'neg_{params.test_file}_{params.constrained_neg_prob}.txt', graphs['test']['neg'], id2entity, id2relation)

    links2subgraphs(adj_list, graphs, params, max_label_value)


def get_kge_embeddings(dataset, kge_model):

    path = './experiments/kge_baselines/{}_{}'.format(kge_model, dataset)
    node_features = np.load(os.path.join(path, 'entity_embedding.npy'))
    with open(os.path.join(path, 'id2entity.json')) as json_file:
        kge_id2entity = json.load(json_file)
        kge_entity2id = {v: int(k) for k, v in kge_id2entity.items()}

    return node_features, kge_entity2id



class SubgraphDataset(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""
    
               
    def __init__(self, db_path, db_name_pos, db_name_neg, raw_data_paths, included_relations=None, add_traspose_rels=False, num_neg_samples_per_link=1, use_kge_embeddings=False, dataset='', kge_model='', file_name='', placn_size=20):

        self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
        self.db_pos = self.main_env.open_db(db_name_pos.encode())
        self.db_neg = self.main_env.open_db(db_name_neg.encode())
        self.node_features, self.kge_entity2id = get_kge_embeddings(dataset, kge_model) if use_kge_embeddings else (None, None)
        self.num_neg_samples_per_link = num_neg_samples_per_link
        self.file_name = file_name
        self.placn_size=placn_size
        ssp_graph, __, __, __, id2entity, id2relation = process_files(raw_data_paths, included_relations)
        self.num_rels = len(ssp_graph)

        # Add transpose matrices to handle both directions of relations.
        if add_traspose_rels:
            ssp_graph_t = [adj.T for adj in ssp_graph]
            ssp_graph += ssp_graph_t
        
        A_incidence = incidence_matrix(ssp_graph)
        A_incidence += A_incidence.T
        # the effective number of relations after adding symmetric adjacency matrices and/or self connections
        self.aug_num_rels = len(ssp_graph)
        self.graph = ssp_multigraph_to_dgl(ssp_graph)
        # compile node features as a 6 dimensional vector
        # [other node][CN][JC][AA][PA][RA]

        n_nodes = self.graph.number_of_nodes();
        #tensor of features to use to look up features by nodes (i, j)
        self.placn_features = np.zeros((n_nodes, n_nodes, 5))
        neighborCache = {}
        for i in tqdm(range(0,n_nodes)):
            if i in neighborCache:
                i_nei = neighborCache[i]
            else:
                i_nei = get_neighbor_nodes(set([i]), A_incidence, 1, None)
                neighborCache[i] = i_nei
            
            for j in range(i,n_nodes): 
            	#pointless to compare to itself
                if i==j: continue
                
                if j in neighborCache:
                    j_nei = neighborCache[j]
                else:
                    j_nei = get_neighbor_nodes(set([j]), A_incidence, 1, None)
                    neighborCache[j] = j_nei

                cn_set = set(i_nei)
                cn_set.intersection_update(set(j_nei))
                self.placn_features[i][j][0] = len(cn_set)#Common neighbours

                all_nei = set(i_nei)
                all_nei.union(set(j_nei))
                self.placn_features[i][j][1] = len(cn_set) / len(all_nei) #Jerard coefficient
                
                aa_sum = 0;#adamic-adair
                for k in all_nei:
                    if k in neighborCache != None:
                        k_nei = neighborCache[k]
                    else:
                        k_nei = get_neighbor_nodes(set([k]), A_incidence, 1, None)
                        neighborCache[k] = k_nei
                    aa_sum = aa_sum + 1/math.log(max(2, len(k_nei)))
                self.placn_features[i][j][2] = aa_sum #adamic-adair

                #preferential attachment
                pa = len(j_nei) * len(i_nei)
                self.placn_features[i][j][3] = pa #

                
                #resource allocation
                #skipped as variation of adamic adair
                ra_sum = 0;
                for k in all_nei:
                    if k in neighborCache != None:
                        k_nei = neighborCache[k]
                    else:
                        k_nei = get_neighbor_nodes(set([k]), A_incidence, 1, None)
                        neighborCache[k] = k_nei
                    ra_sum = ra_sum + 1/len(k_nei)
                self.placn_features[i][j][4] = ra_sum #adamic-adair

                #mirror to lower half
                self.placn_features[j][i] = self.placn_features[i][j]
        self.ssp_graph = ssp_graph
        self.id2entity = id2entity
        self.id2relation = id2relation

        with self.main_env.begin() as txn:
            self.max_n_label = struct.unpack('i', txn.get('max_n_label'.encode()))[0]
            self.avg_subgraph_size = struct.unpack('f', txn.get('avg_subgraph_size'.encode()))
            self.min_subgraph_size = struct.unpack('f', txn.get('min_subgraph_size'.encode()))
            self.max_subgraph_size = struct.unpack('f', txn.get('max_subgraph_size'.encode()))
            self.std_subgraph_size = struct.unpack('f', txn.get('std_subgraph_size'.encode()))

            self.avg_enc_ratio = struct.unpack('f', txn.get('avg_enc_ratio'.encode()))
            self.min_enc_ratio = struct.unpack('f', txn.get('min_enc_ratio'.encode()))
            self.max_enc_ratio = struct.unpack('f', txn.get('max_enc_ratio'.encode()))
            self.std_enc_ratio = struct.unpack('f', txn.get('std_enc_ratio'.encode()))

            self.avg_num_pruned_nodes = struct.unpack('f', txn.get('avg_num_pruned_nodes'.encode()))
            self.min_num_pruned_nodes = struct.unpack('f', txn.get('min_num_pruned_nodes'.encode()))
            self.max_num_pruned_nodes = struct.unpack('f', txn.get('max_num_pruned_nodes'.encode()))
            self.std_num_pruned_nodes = struct.unpack('f', txn.get('std_num_pruned_nodes'.encode()))

        logging.info(f"Max distance node label: {self.max_n_label}")

        # logging.info('=====================')
        # logging.info(f"Subgraph size stats: \n Avg size {self.avg_subgraph_size}, \n Min size {self.min_subgraph_size}, \n Max size {self.max_subgraph_size}, \n Std {self.std_subgraph_size}")

        # logging.info('=====================')
        # logging.info(f"Enclosed nodes ratio stats: \n Avg size {self.avg_enc_ratio}, \n Min size {self.min_enc_ratio}, \n Max size {self.max_enc_ratio}, \n Std {self.std_enc_ratio}")

        # logging.info('=====================')
        # logging.info(f"# of pruned nodes stats: \n Avg size {self.avg_num_pruned_nodes}, \n Min size {self.min_num_pruned_nodes}, \n Max size {self.max_num_pruned_nodes}, \n Std {self.std_num_pruned_nodes}")

        with self.main_env.begin(db=self.db_pos) as txn:
            self.num_graphs_pos = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
        with self.main_env.begin(db=self.db_neg) as txn:
            self.num_graphs_neg = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')

        self.__getitem__(0)

    def __getitem__(self, index):
        with self.main_env.begin(db=self.db_pos) as txn:
            str_id = '{:08}'.format(index).encode('ascii')
            nodes_pos, r_label_pos, g_label_pos, n_labels_pos = deserialize(txn.get(str_id)).values()
            subgraph_pos = self._prepare_subgraphs(nodes_pos, r_label_pos, n_labels_pos)
        subgraphs_neg = []
        r_labels_neg = []
        g_labels_neg = []
        with self.main_env.begin(db=self.db_neg) as txn:
            for i in range(self.num_neg_samples_per_link):
                str_id = '{:08}'.format(index + i * (self.num_graphs_pos)).encode('ascii')
                nodes_neg, r_label_neg, g_label_neg, n_labels_neg = deserialize(txn.get(str_id)).values()
                subgraphs_neg.append(self._prepare_subgraphs(nodes_neg, r_label_neg, n_labels_neg))
                r_labels_neg.append(r_label_neg)
                g_labels_neg.append(g_label_neg)

        return subgraph_pos, g_label_pos, r_label_pos, subgraphs_neg, g_labels_neg, r_labels_neg

    def __len__(self):
        return self.num_graphs_pos

    def _prepare_subgraphs(self, nodes, r_label, n_labels):
        
        subgraph = dgl.DGLGraph(self.graph.subgraph(nodes))
        subgraph.edata['type'] = self.graph.edata['type'][self.graph.subgraph(nodes).parent_eid]
        subgraph.edata['label'] = torch.tensor(r_label * np.ones(subgraph.edata['type'].shape), dtype=torch.long)

        edges_btw_roots = subgraph.edge_id(0, 1)
        rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == r_label)
        if rel_link.squeeze().nelement() == 0:
            subgraph.add_edge(0, 1)
            subgraph.edata['type'][-1] = torch.tensor(r_label).type(torch.LongTensor)
            subgraph.edata['label'][-1] = torch.tensor(r_label).type(torch.LongTensor)

        # map the id read by GraIL to the entity IDs as registered by the KGE embeddings
        kge_nodes = [self.kge_entity2id[self.id2entity[n]] for n in nodes] if self.kge_entity2id else None
        n_feats = self.node_features[kge_nodes] if self.node_features is not None else None
        subgraph = self._prepare_features_placn(nodes, subgraph, n_labels)

        return subgraph

    def _prepare_features_placn(self, nodes, subgraph, n_labels):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes,self.placn_size))
        label_feats[np.array(np.arange(n_nodes)), n_labels] = 1
        placn_subfeats=np.zeros((n_nodes, self.placn_size * 5))
        #Reason to start at, from grail paper:
        #We always assign zero to the positive target link in the adjacency matrix
#of the weighted graph. The reason is that when we test PLACN
#model, positive links should not contain any information of the link’s
#existence. PLACN needs to learn both the positive and negative links
#without the links’ existing information.
        for i in range(2, n_nodes):
            for j in range(2, n_nodes):
                for f in range(0, 5):
                    placn_subfeats[i][5*j + f] = self.placn_features[nodes[i]][nodes[j]][f]
        n_feats = np.concatenate((label_feats,placn_subfeats), axis=1) 
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

        head_id = np.argwhere([label == 0 for label in n_labels])
        tail_id = np.argwhere([label == 1 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)

        self.n_feat_dim = n_feats.shape[1]  
        if self.n_feat_dim > self.placn_size*6:
            print(nodes)
            print(self.n_feat_dim)
            die()
        return subgraph
