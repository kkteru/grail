# #!/usr/bin/env bash

DATA_DIR="../data"
TRAIN_DIR="$DATA_DIR/$1"

TRAIN_FILE="train.txt"

LEARNED_RULES="learned_rules.txt"

#Learn rules
java -cp RuleN.jar de.unima.ki.arch.LearnRules -t $TRAIN_DIR/$TRAIN_FILE -s1 1000 -s2 1000 -p 4 -o $TRAIN_DIR/$LEARNED_RULES
