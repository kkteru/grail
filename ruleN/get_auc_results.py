import os
import argparse

import numpy as np
from sklearn import metrics


def main(params):
    pos_result_file = os.path.join(params.data_dir, 'test_predictions_results.npy')
    neg_result_file = os.path.join(params.data_dir, 'neg_test_0_predictions_results.npy')

    print('Reading positive and negative result files from ', params.data_dir)

    pos_results = np.load(pos_result_file)
    neg_results = np.load(neg_result_file)

    # print(len(pos_results.item()['head']), len(neg_results.item()['head']))
    # # Not necessary since some relations would be filtered out if not preesent during training phase.
    # assert len(pos_results.item()['head']) == len(neg_results.item()['head'])

    all_labels = [1] * 2 * len(pos_results.item()['head']) + [0] * 2 * len(neg_results.item()['head'])
    all_scores = pos_results.item()['head'][:, 1].tolist() + \
        pos_results.item()['tail'][:, 1].tolist() + \
        neg_results.item()['head'][:, 1].tolist() + \
        neg_results.item()['tail'][:, 1].tolist()

    auc = metrics.roc_auc_score(all_labels, all_scores)
    auc_pr = metrics.average_precision_score(all_labels, all_scores)

    rankList = pos_results.item()['head'][:, 0].tolist() + pos_results.item()['tail'][:, 0].tolist()

    mr = np.mean(rankList)
    mrr = np.mean(1 / np.array(rankList))
    hit10 = len([x for x in rankList if x <= 10]) / len(rankList)

    with open(os.path.join(params.data_dir, 'ruleN_test_auc.txt'), "w") as f:
        f.write('AUC_PR score : %f, AUC score : %f,MR : %f, MRR : %f, Hits@10 : %f\n' % (auc_pr, auc, mr, mrr, hit10))
    print('AUC_PR score : %f, AUC score : %f,MR : %f, MRR : %f, Hits@10 : %f' % (auc_pr, auc, mr, mrr, hit10))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TransE model')

    parser.add_argument("--dataset", "-d", type=str, default="tmp1",
                        help="Dataset string")

    params = parser.parse_args()

    params.data_dir = os.path.join(os.path.relpath(os.path.dirname(os.path.abspath(__file__))), '../data/') + params.dataset

    main(params)
