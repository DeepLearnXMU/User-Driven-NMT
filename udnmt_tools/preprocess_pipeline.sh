#!/usr/bin/env bash
# Author : Thamme Gowda
# Created : Nov 06, 2017

# bpe & build_vocab
ONMT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

#======= EXPERIMENT SETUP ======
# Activate python environment if needed
source ~/.bashrc
# source activate py3

BPE="" # default
BPE="src+tgt" # src, tgt, src+tgt

# applicable only when BPE="src" or "src+tgt"
BPE_SRC_OPS=32000

# applicable only when BPE="tgt" or "src+tgt"
BPE_TGT_OPS=32000

GPUARG="" # default
GPUARG="0"

#====== EXPERIMENT BEGIN ======
# wmt17 raw data
DATA="$ONMT/data/wmt17"
TRAIN_SRC=$DATA/src-train.txt
TRAIN_TGT=$DATA/tgt-train.txt
VALID_SRC=$DATA/src-val.txt
VALID_TGT=$DATA/tgt-val.txt
TEST_SRC=$DATA/src-test.txt
TEST_TGT=$DATA/src-test.txt

# Check if input exists
for f in $TRAIN_SRC $TRAIN_TGT $VALID_SRC $VALID_TGT $TEST_SRC $TEST_TGT; do
    if [[ ! -f "$f" ]]; then
        echo "Input File $f doesnt exist. Please fix the paths"
        exit 1
    fi
done

function lines_check {
    l1=$(wc -l $1)
    l2=$(wc -l $2)
    if [[ ${l1% *} != ${l2% *} ]]; then
        echo $l1
        echo $l2
        echo "ERROR: Record counts doesnt match between: $1 and $2"
        exit 2
    fi
}
lines_check $TRAIN_SRC $TRAIN_TGT
lines_check $VALID_SRC $VALID_TGT
lines_check $TEST_SRC $TEST_TGT

echo "Step 1a: Preprocess inputs (WMT17)"
if [[ "$BPE" == *"src"* ]]; then
    echo "BPE on source"
    # Here we could use more monolingual data
    $ONMT/tools/learn_bpe.py -s $BPE_SRC_OPS < $TRAIN_SRC > $DATA/bpe-codes.src
    $ONMT/tools/apply_bpe.py -c $DATA/bpe-codes.src <  $TRAIN_SRC > $DATA/train.src
    $ONMT/tools/apply_bpe.py -c $DATA/bpe-codes.src <  $VALID_SRC > $DATA/valid.src
    $ONMT/tools/apply_bpe.py -c $DATA/bpe-codes.src <  $TEST_SRC > $DATA/test.src
else
    ln -sf $TRAIN_SRC $DATA/train.src
    ln -sf $VALID_SRC $DATA/valid.src
    ln -sf $TEST_SRC $DATA/test.src
fi

if [[ "$BPE" == *"tgt"* ]]; then
    echo "BPE on target (WMT17)"
    # Here we could use more  monolingual data
    $ONMT/tools/learn_bpe.py -s $BPE_SRC_OPS < $TRAIN_TGT > $DATA/bpe-codes.tgt
    $ONMT/tools/apply_bpe.py -c $DATA/bpe-codes.tgt <  $TRAIN_TGT > $DATA/train.tgt
    # We dont touch the test and valid References, No BPE on them!
    ln -sf $VALID_TGT $DATA/valid.tgt
    ln -sf $TEST_TGT $DATA/test.tgt
else
    ln -sf $TRAIN_TGT $DATA/train.tgt
    ln -sf $VALID_TGT $DATA/valid.tgt
    ln -sf $TEST_TGT $DATA/test.tgt
fi

# udt-corpus raw data
DATA="$ONMT/data/user"
TRAIN_SRC=$DATA/train.zh
TRAIN_TGT=$DATA/train.en
VALID_SRC=$DATA/valid.zh
VALID_TGT=$DATA/valid.en
TEST_SRC=$DATA/test.zh
TEST_TGT=$DATA/test.en

# Check if input exists
for f in $TRAIN_SRC $TRAIN_TGT $VALID_SRC $VALID_TGT $TEST_SRC $TEST_TGT; do
    if [[ ! -f "$f" ]]; then
        echo "Input File $f doesnt exist. Please fix the paths"
        exit 1
    fi
done

function lines_check {
    l1=$(wc -l $1)
    l2=$(wc -l $2)
    if [[ ${l1% *} != ${l2% *} ]]; then
        echo $l1
        echo $l2
        echo "ERROR: Record counts doesnt match between: $1 and $2"
        exit 2
    fi
}
lines_check $TRAIN_SRC $TRAIN_TGT
lines_check $VALID_SRC $VALID_TGT
lines_check $TEST_SRC $TEST_TGT

echo "Step 1b: Preprocess inputs (UDT-Corpus)"
if [[ "$BPE" == *"src"* ]]; then
    echo "BPE on source"
    # Here we could use more monolingual data
    for i in train valid test
    do
        $ONMT/tools/apply_bpe.py -c $DATA/../wmt17/bpe-codes.src <  $DATA/$i.zh > $DATA/$i.src
        $ONMT/tools/apply_bpe.py -c $DATA/../wmt17/bpe-codes.src <  $DATA/cache/topic25.$i > $DATA/cache/topic25.$i.src
        $ONMT/tools/apply_bpe.py -c $DATA/../wmt17/bpe-codes.src <  $DATA/cache/context35.$i > $DATA/cache/context35.$i.src
    done
    $ONMT/tools/apply_bpe.py -c $DATA/../wmt17/bpe-codes.src <  $DATA/cache/far25.train > $DATA/cache/far25.train.src
    $ONMT/tools/apply_bpe.py -c $DATA/../wmt17/bpe-codes.src <  $DATA/cache/nearby25.train > $DATA/cache/nearby25.train.src
else
    ln -sf $TRAIN_SRC $DATA/train.src
    ln -sf $VALID_SRC $DATA/valid.src
    ln -sf $TEST_SRC $DATA/test.src
fi

if [[ "$BPE" == *"tgt"* ]]; then
    echo "BPE on target (UDT-Corpus)"
    # Here we could use more  monolingual data
    $ONMT/tools/apply_bpe.py -c $DATA/../wmt17/bpe-codes.tgt <  $TRAIN_TGT > $DATA/train.tgt
    # We dont touch the test and valid References, No BPE on them!
    ln -sf $VALID_TGT $DATA/valid.tgt
    ln -sf $TEST_TGT $DATA/test.tgt
else
    ln -sf $TRAIN_TGT $DATA/train.tgt
    ln -sf $VALID_TGT $DATA/valid.tgt
    ln -sf $TEST_TGT $DATA/test.tgt
fi

#: <<EOF
echo "Step 2: Build Vocab"
python $ONMT/onmt/bin/build_vocab.py \
    -config $ONMT/config/vocab.yml

#===== EXPERIMENT END ======

