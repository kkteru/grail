#!/usr/bin/env bash

# This script assumes GraIL predection scores on the validation and test set are already saved.
# It also assumes that scored head/tail replaced triplets are also stored.
# If any of those is not present, run the corresponding script from the following setup commands.
##################### SET UP #####################
# python test_auc.py -d WN18RR -e saved_grail_exp_name --hop 3 -t valid
# python test_auc.py -d WN18RR -e saved_grail_exp_name --hop 3 -t test

# python test_auc.py -d NELL-995 -e saved_grail_exp_name --hop 2 -t valid
# python test.py -d NELL-995 -e saved_grail_exp_name --hop 2 -t test

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
KGE_MODEL=$2
KGE_SAVED_MODEL_PATH="../experiments/kge_baselines/${KGE_MODEL}_${DATASET}"

# score pos validation triplets with KGE model
python score_triplets_kge.py -d $DATASET --model $KGE_MODEL -f valid -init $KGE_SAVED_MODEL_PATH
# score neg validation triplets with KGE model
python score_triplets_kge.py -d $DATASET --model $KGE_MODEL -f neg_valid_0 -init $KGE_SAVED_MODEL_PATH

# train the ensemble model
python blend.py -d $DATASET -em2 $KGE_MODEL --do_train -ne 500

# Score the test pos and neg triplets with KGE model
python score_triplets_kge.py -d $DATASET --model $KGE_MODEL -f test -init $KGE_SAVED_MODEL_PATH
python score_triplets_kge.py -d $DATASET --model $KGE_MODEL -f neg_test_0 -init $KGE_SAVED_MODEL_PATH
# Score the test pos and neg triplets with ensemble model
python blend.py -d $DATASET -em2 $KGE_MODEL --do_scoring -f test
python blend.py -d $DATASET -em2 $KGE_MODEL --do_scoring -f neg_test_0
# Compute auc with the ensemble model scored pos and neg test files
python compute_auc.py -d $DATASET -m grail_with_${KGE_MODEL}
# Compute auc with the KGE model model scored pos and neg test files
python compute_auc.py -d $DATASET -m $KGE_MODEL

# Score head/tail replaced samples with KGE model
python score_triplets_kge.py -d $DATASET --model $KGE_MODEL -f ranking_head -init $KGE_SAVED_MODEL_PATH
python score_triplets_kge.py -d $DATASET --model $KGE_MODEL -f ranking_tail -init $KGE_SAVED_MODEL_PATH
# Score head/tail replaced samples with ensemble model
python blend.py -d $DATASET -em2 $KGE_MODEL --do_scoring -f ranking_head
python blend.py -d $DATASET -em2 $KGE_MODEL --do_scoring -f ranking_tail
# Compute ranking metrics for ensemble model with the scored head/tail replaced samples
python compute_rank_metrics.py -d $DATASET -m grail_with_${KGE_MODEL}
# Compute ranking metrics for KGE model with the scored head/tail replaced samples
python compute_rank_metrics.py -d $DATASET -m $KGE_MODEL