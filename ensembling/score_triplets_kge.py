#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(1, '../')

import argparse
import json
import logging
import os

import torch

from kge.model import KGEModel

from utils.data_utils import process_files


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')

    parser.add_argument('--dataset', '-d', type=str, default='Toy')
    parser.add_argument('--model', '-m', default='TransE', type=str)
    parser.add_argument('--file_to_score', '-f', default='test', type=str)
    parser.add_argument('--init_checkpoint', '-init', default=None, type=str)

    return parser.parse_args(args)


def override_config(args):
    '''
    Override model and data configuration
    '''

    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)

    args.countries = argparse_dict['countries']
    if args.dataset is None:
        args.dataset = argparse_dict['dataset']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']
    args.gamma = argparse_dict['gamma']


def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''
    log_file = os.path.join(args.init_checkpoint, 'score_{}.log'.format(args.file_to_score))

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def main(args):
    if args.init_checkpoint:
        override_config(args)
    elif args.dataset is None:
        raise ValueError('one of init_checkpoint/dataset must be choosed.')

    # Write logs to checkpoint and console
    set_logger(args)

    main_dir = os.path.join(os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.')

    with open(os.path.join(main_dir, 'data/{}/entities.dict'.format(args.dataset))) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(main_dir, 'data/{}/relations.dict'.format(args.dataset))) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    # test_triples = to_kge_format(triplets['to_score'])
    test_triples = read_triple(os.path.join(main_dir, 'data/{}/{}.txt'.format(args.dataset, args.file_to_score)), entity2id, relation2id)

    nentity = len(entity2id)
    nrelation = len(relation2id)
    args.nentity = nentity
    args.nrelation = nrelation

    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.dataset)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)

    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding
    )

    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()

    # Restore model from checkpoint directory
    logging.info('Loading checkpoint %s...' % args.init_checkpoint)
    checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
    kge_model.load_state_dict(checkpoint['model_state_dict'])
    logging.info('Scoring the triplets in {}.txt file'.format(args.file_to_score))
    scores = kge_model.score_triplets(kge_model, test_triples, args)

    with open(os.path.join(main_dir, 'data/{}/{}.txt'.format(args.dataset, args.file_to_score))) as f:
        triplets = [line.split() for line in f.read().split('\n')[:-1]]
    file_path = os.path.join(main_dir, 'data/{}/{}_{}_predictions.txt'.format(args.dataset, args.model, args.file_to_score))
    with open(file_path, "w") as f:
        for ([s, r, o], score) in zip(triplets, scores):
            f.write('\t'.join([s, r, o, str(score)]) + '\n')


if __name__ == '__main__':
    main(parse_args())
