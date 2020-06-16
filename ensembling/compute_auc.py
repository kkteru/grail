import argparse
from sklearn import metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute AUC from scored positive and negative triplets')

    parser.add_argument('--dataset', '-d', type=str, default='Toy')
    parser.add_argument('--model', '-m', default='ens', type=str)
    parser.add_argument('--test_file', '-t', default='test', type=str)

    params = parser.parse_args()

    # load pos and neg prediction scores of the test_file of the dataset for the given model.
    with open('../data/{}/{}_{}_predictions.txt'.format(params.dataset, params.model, params.test_file)) as f:
        pos_scores = [float(line.split()[-1]) for line in f.read().split('\n')[:-1]]
    with open('../data/{}/{}_neg_{}_0_predictions.txt'.format(params.dataset, params.model, params.test_file)) as f:
        neg_scores = [float(line.split()[-1]) for line in f.read().split('\n')[:-1]]

    # compute auc score
    scores = pos_scores + neg_scores
    labels = [1] * len(pos_scores) + [0] * len(neg_scores)

    auc = metrics.roc_auc_score(labels, scores)
    auc_pr = metrics.average_precision_score(labels, scores)

    with open('../data/{}/{}_{}_auc.txt'.format(params.dataset, params.model, params.test_file), "w") as f:
        f.write('AUC : {}, AUC_PR : {}\n'.format(auc, auc_pr))
