#!/usr/bin/env bash
# Author : Thamme Gowda
# Created : Nov 06, 2017
#train, valid and test

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#======= EXPERIMENT SETUP ======
# Activate python environment if needed
source ~/.bashrc
# source activate py

cat $DIR/train.bpe.en | sed -E 's/(@@ )|(@@ ?$)//g' > $DIR/train.en
cat $DIR/test.bpe.en | sed -E 's/(@@ )|(@@ ?$)//g' > $DIR/test.en
cat $DIR/dev.bpe.en | sed -E 's/(@@ )|(@@ ?$)//g' > $DIR/dev.en
cat $DIR/train.bpe.zh | sed -E 's/(@@ )|(@@ ?$)//g' > $DIR/train.zh
cat $DIR/test.bpe.zh | sed -E 's/(@@ )|(@@ ?$)//g' > $DIR/test.zh
cat $DIR/dev.bpe.zh | sed -E 's/(@@ )|(@@ ?$)//g' > $DIR/dev.zh

#===== EXPERIMENT END ======

