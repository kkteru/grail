# #!/usr/bin/env bash

DATA_DIR="../data"
TRAIN_DIR="$DATA_DIR/$1"
TEST_DIR="$DATA_DIR/$2"

TRAIN_FILE="train.txt"
VALID_FILE="valid.txt"
HEAD_TEST_FILE="ranking_head.txt"
TAIL_TEST_FILE="ranking_tail.txt"
LEARNED_RULES="learned_rules.txt"
HEAD_PRED="ranking_head_predictions.txt"
TAIL_PRED="ranking_tail_predictions.txt"

# Apply rules to head replaced neg samples
java -cp RuleN.jar de.unima.ki.arch.ApplyRules -tr $TEST_DIR/$TRAIN_FILE -v $TEST_DIR/$VALID_FILE -t $TEST_DIR/$HEAD_TEST_FILE -r $TRAIN_DIR/$LEARNED_RULES -o $TEST_DIR/$HEAD_PRED -k $3

# Apply rules to tail replaced neg samples
java -cp RuleN.jar de.unima.ki.arch.ApplyRules -tr $TEST_DIR/$TRAIN_FILE -v $TEST_DIR/$VALID_FILE -t $TEST_DIR/$TAIL_TEST_FILE -r $TRAIN_DIR/$LEARNED_RULES -o $TEST_DIR/$TAIL_PRED -k $3

# Process both head and tail predictions to get rank and scores
python process_predictions.py -d $2 -f $HEAD_PRED
python process_predictions.py -d $2 -f $TAIL_PRED

# Process both +ve and -ve predictions to get rank and scores
python get_ranking_results.py -d $2 
