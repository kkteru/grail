import argparse
import os


import torch
import torch.nn as nn
import torch.optim as optim


def read_scores(path):
    with open(path) as f:
        scores = [float(line.split()[-1]) for line in f.read().split('\n')[:-1]]
    return scores


def get_triplets(path):
    with open(path) as f:
        triplets = [line.split()[:-1] for line in f.read().split('\n')[:-1]]
    return triplets


def train(params):
    '''
    Train and save a linear layer model.
    '''
    ens_model_1_pos_scores_path = os.path.join('../data/{}/{}_valid_predictions.txt'.format(params.dataset, params.ensemble_model_1))
    ens_model_1_neg_scores_path = os.path.join('../data/{}/{}_neg_valid_0_predictions.txt'.format(params.dataset, params.ensemble_model_1))
    ens_model_2_pos_scores_path = os.path.join('../data/{}/{}_valid_predictions.txt'.format(params.dataset, params.ensemble_model_2))
    ens_model_2_neg_scores_path = os.path.join('../data/{}/{}_neg_valid_0_predictions.txt'.format(params.dataset, params.ensemble_model_2))

    assert get_triplets(ens_model_1_pos_scores_path) == get_triplets(ens_model_2_pos_scores_path)
    assert get_triplets(ens_model_1_neg_scores_path) == get_triplets(ens_model_2_neg_scores_path)

    pos_scores = torch.Tensor(list(zip(read_scores(ens_model_1_pos_scores_path), read_scores(ens_model_2_pos_scores_path))))
    neg_scores = torch.Tensor(list(zip(read_scores(ens_model_1_neg_scores_path), read_scores(ens_model_2_neg_scores_path))))

    # scores = pos_scores + neg_scores
    # targets = [1] * len(pos_scores) + [0] * len(neg_scores)

    model = nn.Linear(in_features=2, out_features=1)
    criterion = nn.MarginRankingLoss(10, reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)

    for e in range(params.num_epochs):
        pos_out = model(pos_scores)
        neg_out = model(neg_scores)

        loss = criterion(pos_out, neg_out.view(len(pos_out), -1).mean(dim=1), torch.Tensor([1]))
        print('Loss at epoch {} : {}'.format(e, loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model, os.path.join('../experiments', f'{params.ensemble_model_1}_{params.ensemble_model_2}_{params.dataset}_ensemble.pth'))


def score_triplets(params):
    '''
    Load the saved model and save scores of given set of triplets.
    '''
    print('Loading model..')
    model = torch.load(os.path.join('../experiments', f'{params.ensemble_model_1}_{params.ensemble_model_2}_{params.dataset}_ensemble.pth'))
    print('Model loaded successfully!')

    ens_model_1_scores_path = os.path.join('../data/{}/{}_{}_predictions.txt'.format(params.dataset, params.ensemble_model_1, params.file_to_score))
    ens_model_2_scores_path = os.path.join('../data/{}/{}_{}_predictions.txt'.format(params.dataset, params.ensemble_model_2, params.file_to_score))

    scores = torch.Tensor(list(zip(read_scores(ens_model_1_scores_path), read_scores(ens_model_2_scores_path))))
    ens_scores = model(scores)

    ens_model_1_triplets = get_triplets(ens_model_1_scores_path)
    ens_model_2_triplets = get_triplets(ens_model_2_scores_path)

    assert ens_model_1_triplets == ens_model_2_triplets

    file_path = os.path.join('../', 'data/{}/{}_with_{}_{}_predictions.txt'.format(params.dataset, params.ensemble_model_1, params.ensemble_model_2, params.file_to_score))
    with open(file_path, "w") as f:
        for ([s, r, o], score) in zip(ens_model_1_triplets, ens_scores):
            f.write('\t'.join([s, r, o, str(score.item())]) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model blender script')

    parser.add_argument('--dataset', '-d', type=str, default='Toy')
    parser.add_argument('--ensemble_model_1', '-em1', default='grail', type=str)
    parser.add_argument('--ensemble_model_2', '-em2', default='TransE', type=str)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument("--num_epochs", "-ne", type=int, default=500,
                        help="Number of training iterations")
    parser.add_argument('--do_scoring', action='store_true')
    parser.add_argument('--file_to_score', '-f', default='valid', type=str)

    params = parser.parse_args()

    if params.do_train:
        train(params)
    elif params.do_scoring:
        score_triplets(params)
