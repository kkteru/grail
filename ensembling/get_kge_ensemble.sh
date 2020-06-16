#!/usr/bin/env bash

# This script assumes that head/tail replaced negative triplets are already stored while evaluating GraIL.
# This assumptionn is made in order to make fair evaluations of all the methods on the same negative samples.
# If any of those is not present, run the corresponding script from the following setup commands. These will
# evaluate GraIL and savee thee negative samples along the way.
##################### SET UP #####################
# python test_auc.py -d WN18RR -e saved_grail_exp_name --hop 3 -t valid
# python test_auc.py -d WN18RR -e saved_grail_exp_name --hop 3 -t test

# python test_auc.py -d NELL-995 -e saved_grail_exp_name --hop 2 -t valid
# python test_auc.py -d NELL-995 -e saved_grail_exp_name --hop 2 -t test

# python test_auc.py -d FB15K237 -e saved_grail_exp_name --hop 1 -t valid
# python test_auc.py -d FB15K237 -e saved_grail_exp_name --hop 1 -t test

# python test_ranking.py -d WN18RR -e saved_grail_exp_name --hop 3

# python test_ranking.py -d NELL-995 -e saved_grail_exp_name --hop 2

# python test_ranking.py -d FB15K237 -e saved_grail_exp_name --hop 1
##################################################


# Arguments
# Dataset
DATASET=$1
# KGE model to be used in ensemble
KGE_MODEL_1=$2
KGE_SAVED_MODEL_PATH_1="../experiments/kge_baselines/${KGE_MODEL_1}_${DATASET}"

KGE_MODEL_2=$3
KGE_SAVED_MODEL_PATH_2="../experiments/kge_baselines/${KGE_MODEL_2}_${DATASET}"

# score pos validation triplets with KGE model
python score_triplets_kge.py -d $DATASET --model $KGE_MODEL_1 -f valid -init $KGE_SAVED_MODEL_PATH_1
# score neg validation triplets with KGE model
python score_triplets_kge.py -d $DATASET --model $KGE_MODEL_1 -f neg_valid_0 -init $KGE_SAVED_MODEL_PATH_1

# score pos validation triplets with KGE model
python score_triplets_kge.py -d $DATASET --model $KGE_MODEL_2 -f valid -init $KGE_SAVED_MODEL_PATH_2
# score neg validation triplets with KGE model
python score_triplets_kge.py -d $DATASET --model $KGE_MODEL_2 -f neg_valid_0 -init $KGE_SAVED_MODEL_PATH_2

# train the ensemble model
python blend.py -d $DATASET -em1 $KGE_MODEL_1 -em2 $KGE_MODEL_2 --do_train -ne 500

# Score the test pos and neg triplets with KGE model
python score_triplets_kge.py -d $DATASET --model $KGE_MODEL_1 -f test -init $KGE_SAVED_MODEL_PATH_1
python score_triplets_kge.py -d $DATASET --model $KGE_MODEL_1 -f neg_test_0 -init $KGE_SAVED_MODEL_PATH_1
# Score the test pos and neg triplets with KGE model
python score_triplets_kge.py -d $DATASET --model $KGE_MODEL_2 -f test -init $KGE_SAVED_MODEL_PATH_2
python score_triplets_kge.py -d $DATASET --model $KGE_MODEL_2 -f neg_test_0 -init $KGE_SAVED_MODEL_PATH_2


# Score the test pos and neg triplets with ensemble model
python blend.py -d $DATASET -em1 $KGE_MODEL_1 -em2 $KGE_MODEL_2 --do_scoring -f test
python blend.py -d $DATASET -em1 $KGE_MODEL_1 -em2 $KGE_MODEL_2 --do_scoring -f neg_test_0
# Compute auc with the ensemble model scored pos and neg test files
python compute_auc.py -d $DATASET -m ${KGE_MODEL_1}_with_${KGE_MODEL_2}

# Score head/tail replaced samples with KGE model
python score_triplets_kge.py -d $DATASET --model $KGE_MODEL_1 -f ranking_head -init $KGE_SAVED_MODEL_PATH_1
python score_triplets_kge.py -d $DATASET --model $KGE_MODEL_1 -f ranking_tail -init $KGE_SAVED_MODEL_PATH_1
# Score head/tail replaced samples with KGE model
python score_triplets_kge.py -d $DATASET --model $KGE_MODEL_2 -f ranking_head -init $KGE_SAVED_MODEL_PATH_2
python score_triplets_kge.py -d $DATASET --model $KGE_MODEL_2 -f ranking_tail -init $KGE_SAVED_MODEL_PATH_2


# Score head/tail replaced samples with ensemble model
python blend.py -d $DATASET -em1 $KGE_MODEL_1 -em2 $KGE_MODEL_2 --do_scoring -f ranking_head
python blend.py -d $DATASET -em1 $KGE_MODEL_1 -em2 $KGE_MODEL_2 --do_scoring -f ranking_tail
# Compute ranking metrics for ensemble model with the scored head/tail replaced samples
python compute_rank_metrics.py -d $DATASET -m ${KGE_MODEL_1}_with_${KGE_MODEL_2}