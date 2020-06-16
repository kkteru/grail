import os
import argparse

import numpy as np

TOT_ENTITIES = 40942  # Needs to be a parameter!


def get_evaluations(scores):
    e = []
    for i in range(len(scores) // 2):
        e.append([scores[2 * i], scores[2 * i + 1]])
    return np.array(e)


def save_result(pred, file_name):
    tuples = []
    h_lists = []
    t_lists = []
    for i in range(len(pred) // 3):
        tuples.append([pred[3 * i][0], pred[3 * i][1], pred[3 * i][2]])
        h_lists.append(get_evaluations(pred[3 * i + 1][1:]))
        t_lists.append(get_evaluations(pred[3 * i + 2][1:]))

    h = []
    t = []
    for i, (tple, h_list, t_list) in enumerate(zip(tuples, h_lists, t_lists)):
        h_idx = [] if len(h_list) == 0 else np.argwhere(h_list[:, 0] == tple[0])
        t_idx = [] if len(t_list) == 0 else np.argwhere(t_list[:, 0] == tple[2])

        if len(h_idx) > 0:
            head_rank = h_idx[0][0] + 1
            head_score = float(h_list[h_idx[0][0], 1])
        else:
            head_rank = TOT_ENTITIES // 2
            head_score = 0

        if len(t_idx) > 0:
            tail_rank = t_idx[0][0] + 1
            tail_score = float(t_list[t_idx[0][0], 1])
        else:
            tail_rank = TOT_ENTITIES // 2
            tail_score = 0

        h.append([head_rank, head_score])
        t.append([tail_rank, tail_score])

    results = {}
    results['head'] = np.array(h)
    results['tail'] = np.array(t)

    result_file = os.path.join(params.data_dir, file_name)

    print('Saving ', result_file)
    np.save(result_file, results)

    print('Writing scores to prediction file...')
    file_path = os.path.join(params.data_dir, 'ruleN_' + params.prediction_file)
    with open(file_path, "w") as f:
        for ([s, r, o], score) in zip(tuples, h):
            f.write('\t'.join([s, r, o, str(score[1])]) + '\n')


def main(params):
    pred_file = os.path.join(params.data_dir, params.prediction_file)

    print(f'Reading prediction file {params.prediction_file} from {params.data_dir}')

    with open(pred_file) as f:
        pred = [line.split() for line in f.read().split('\n')]

    save_result(pred, params.prediction_file.split('.')[0] + '_results.npy')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TransE model')

    parser.add_argument("--dataset", "-d", type=str, default="tmp1",
                        help="Dataset string")
    parser.add_argument("--prediction_file", "-f", type=str, default="pos_predictions.txt",
                        help="Dataset string")

    params = parser.parse_args()

    params.data_dir = os.path.join(os.path.relpath(os.path.dirname(os.path.abspath(__file__))), '../data/') + params.dataset

    main(params)
