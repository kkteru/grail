# #!/usr/bin/env bash

DATA_DIR="../data"
TRAIN_DIR="$DATA_DIR/$1"
TEST_DIR="$DATA_DIR/$2"

TRAIN_FILE="train.txt"
VALID_FILE="valid.txt"
TEST_FILE="test.txt"
NEG_TEST_FILE="neg_test_0.txt"
LEARNED_RULES="learned_rules.txt"
POS_PRED="test_predictions.txt"
NEG_PRED="neg_test_0_predictions.txt"

# Apply rules to positive test set
java -cp RuleN.jar de.unima.ki.arch.ApplyRules -tr $TEST_DIR/$TRAIN_FILE -v $TEST_DIR/$VALID_FILE -t $TEST_DIR/$TEST_FILE -r $TRAIN_DIR/$LEARNED_RULES -o $TEST_DIR/$POS_PRED -k $3

# Apply rules to negative test set
java -cp RuleN.jar de.unima.ki.arch.ApplyRules -tr $TEST_DIR/$TRAIN_FILE -v $TEST_DIR/$VALID_FILE -t $TEST_DIR/$NEG_TEST_FILE -r $TRAIN_DIR/$LEARNED_RULES -o $TEST_DIR/$NEG_PRED -k $3

# Process both +ve and -ve predictions to get rank and scores
python process_predictions.py -d $2 -f $POS_PRED
python process_predictions.py -d $2 -f $NEG_PRED

python get_auc_results.py -d $2
