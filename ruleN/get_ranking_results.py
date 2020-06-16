import os
import argparse

import numpy as np
from scipy.stats import rankdata
from sklearn import metrics


def main(params):
    head_result_file = os.path.join(params.data_dir, 'ranking_head_predictions_results.npy')
    tail_result_file = os.path.join(params.data_dir, 'ranking_tail_predictions_results.npy')

    print('Reading positive and negative result files from ', params.data_dir)

    head_results = np.load(head_result_file).item()['head'][:, 1]
    tail_results = np.load(tail_result_file).item()['head'][:, 1]

    # assert len(pos_results.item()['head']) == len(neg_results.item()['head'])

    head_ranks = []
    for i in range(0, len(head_results), params.num_neg_samples):
        if head_results[i] == 0:
            head_ranks.append(params.num_neg_samples)
        else:
            head_ranks.append(params.num_neg_samples - rankdata(head_results[i:i + params.num_neg_samples], method='min')[0] + 1)

    tail_ranks = []
    for i in range(0, len(tail_results), params.num_neg_samples):
        if tail_results[i] == 0:
            tail_ranks.append(params.num_neg_samples)
        else:
            tail_ranks.append(params.num_neg_samples - rankdata(tail_results[i:i + params.num_neg_samples], method='min')[0] + 1)

    rankList = head_ranks + tail_ranks

    mrr = np.mean(1 / np.array(rankList))
    hits_1 = len([x for x in rankList if x <= 1]) / len(rankList)
    hits_5 = len([x for x in rankList if x <= 5]) / len(rankList)
    hits_10 = len([x for x in rankList if x <= 10]) / len(rankList)

    with open(os.path.join(params.data_dir, 'ruleN_ranking_metrics.txt'), "w") as f:
        f.write(f'MRR | Hits@1 | Hits@5 | Hits@10 : {mrr} | {hits_1} | {hits_5} | {hits_10}\n')
    print(f'MRR | Hits@1 | Hits@5 | Hits@10 : {mrr} | {hits_1} | {hits_5} | {hits_10}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TransE model')

    parser.add_argument("--dataset", "-d", type=str, default="tmp1",
                        help="Dataset string")
    parser.add_argument("--num_neg_samples", "-k", type=int, default=50,
                        help="Number of negative samples per test link")

    params = parser.parse_args()

    params.data_dir = os.path.join(os.path.relpath(os.path.dirname(os.path.abspath(__file__))), '../data/') + params.dataset

    main(params)
