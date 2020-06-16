import argparse
import os

import numpy as np
from scipy.stats import rankdata


def get_ranks(scores):
    '''
    Given scores of head/tail substituted triplets, return ranks of each triplet.
    Assumes a fixed number of negative samples (50)
    '''
    ranks = []
    for i in range(len(scores) // 50):
        # rank = np.argwhere(np.argsort(scores[50 * i: 50 * (i + 1)])[::-1] == 0) + 1
        rank = 50 - rankdata(scores[50 * i: 50 * (i + 1)], method='min')[0] + 1
        ranks.append(rank)
    return ranks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute AUC from scored positive and negative triplets')

    parser.add_argument('--dataset', '-d', type=str, default='Toy')
    parser.add_argument('--model', '-m', default='ens', type=str)

    params = parser.parse_args()

    # load head and tail prediction scores of the test file of the dataset for the given model.
    with open('../data/{}/{}_ranking_head_predictions.txt'.format(params.dataset, params.model)) as f:
        head_scores = [float(line.split()[-1]) for line in f.read().split('\n')[:-1]]
    with open('../data/{}/{}_ranking_tail_predictions.txt'.format(params.dataset, params.model)) as f:
        tail_scores = [float(line.split()[-1]) for line in f.read().split('\n')[:-1]]

    # compute both ranks from the prediction scores
    head_ranks = get_ranks(head_scores)
    tail_ranks = get_ranks(tail_scores)

    ranks = head_ranks + tail_ranks

    isHit1List = [x for x in ranks if x <= 1]
    isHit5List = [x for x in ranks if x <= 5]
    isHit10List = [x for x in ranks if x <= 10]
    hits_1 = len(isHit1List) / len(ranks)
    hits_5 = len(isHit5List) / len(ranks)
    hits_10 = len(isHit10List) / len(ranks)

    mrr = np.mean(1 / np.array(ranks))

    with open('../data/{}/{}_ranking_metrics.txt'.format(params.dataset, params.model), "w") as f:
        f.write(f'MRR | Hits@1 | Hits@5 | Hits@10 : {mrr} | {hits_1} | {hits_5} | {hits_10}\n')
