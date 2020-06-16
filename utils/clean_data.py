import os
import argparse
import numpy as np


def write_to_file(file_name, data):
    with open(file_name, "w") as f:
        for s, r, o in data:
            f.write('\t'.join([s, r, o]) + '\n')


def main(params):
    with open(os.path.join(params.main_dir, 'data', params.dataset, 'train.txt')) as f:
        train_data = [line.split() for line in f.read().split('\n')[:-1]]
    with open(os.path.join(params.main_dir, 'data', params.dataset, 'valid.txt')) as f:
        valid_data = [line.split() for line in f.read().split('\n')[:-1]]
    with open(os.path.join(params.main_dir, 'data', params.dataset, 'test.txt')) as f:
        test_data = [line.split() for line in f.read().split('\n')[:-1]]

    train_tails = set([d[2] for d in train_data])
    train_heads = set([d[0] for d in train_data])
    train_ent = train_tails.union(train_heads)
    train_rels = set([d[1] for d in train_data])

    filtered_valid_data = []
    for d in valid_data:
        if d[0] in train_ent and d[1] in train_rels and d[2] in train_ent:
            filtered_valid_data.append(d)
        else:
            train_data.append(d)
            train_ent = train_ent.union(set([d[0], d[2]]))
            train_rels = train_rels.union(set([d[1]]))

    filtered_test_data = []
    for d in test_data:
        if d[0] in train_ent and d[1] in train_rels and d[2] in train_ent:
            filtered_test_data.append(d)
        else:
            train_data.append(d)
            train_ent = train_ent.union(set([d[0], d[2]]))
            train_rels = train_rels.union(set([d[1]]))

    data_dir = os.path.join(params.main_dir, 'data/{}'.format(params.dataset))
    write_to_file(os.path.join(data_dir, 'train.txt'), train_data)
    write_to_file(os.path.join(data_dir, 'valid.txt'), filtered_valid_data)
    write_to_file(os.path.join(data_dir, 'test.txt'), filtered_test_data)

    with open(os.path.join(params.main_dir, 'data', params.dataset + '_meta', 'train.txt')) as f:
        meta_train_data = [line.split() for line in f.read().split('\n')[:-1]]
    with open(os.path.join(params.main_dir, 'data', params.dataset + '_meta', 'valid.txt')) as f:
        meta_valid_data = [line.split() for line in f.read().split('\n')[:-1]]
    with open(os.path.join(params.main_dir, 'data', params.dataset + '_meta', 'test.txt')) as f:
        meta_test_data = [line.split() for line in f.read().split('\n')[:-1]]

    meta_train_tails = set([d[2] for d in meta_train_data])
    meta_train_heads = set([d[0] for d in meta_train_data])
    meta_train_ent = meta_train_tails.union(meta_train_heads)
    meta_train_rels = set([d[1] for d in meta_train_data])

    filtered_meta_valid_data = []
    for d in meta_valid_data:
        if d[0] in meta_train_ent and d[1] in meta_train_rels and d[2] in meta_train_ent:
            filtered_meta_valid_data.append(d)
        else:
            meta_train_data.append(d)
            meta_train_ent = meta_train_ent.union(set([d[0], d[2]]))
            meta_train_rels = meta_train_rels.union(set([d[1]]))

    filtered_meta_test_data = []
    for d in meta_test_data:
        if d[0] in meta_train_ent and d[1] in meta_train_rels and d[2] in meta_train_ent:
            filtered_meta_test_data.append(d)
        else:
            meta_train_data.append(d)
            meta_train_ent = meta_train_ent.union(set([d[0], d[2]]))
            meta_train_rels = meta_train_rels.union(set([d[1]]))

    meta_data_dir = os.path.join(params.main_dir, 'data/{}_meta'.format(params.dataset))
    write_to_file(os.path.join(meta_data_dir, 'train.txt'), meta_train_data)
    write_to_file(os.path.join(meta_data_dir, 'valid.txt'), filtered_meta_valid_data)
    write_to_file(os.path.join(meta_data_dir, 'test.txt'), filtered_meta_test_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Move new entities from test/valid to train')

    parser.add_argument("--dataset", "-d", type=str, default="fb237_v1_copy",
                        help="Dataset string")
    params = parser.parse_args()

    params.main_dir = os.path.join(os.path.relpath(os.path.dirname(os.path.abspath(__file__))), '..')

    main(params)
