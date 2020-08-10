#!/bin/bash

# Copyright 2018 AIShell-Foundation(Authors:Jiayu DU, Xingyu NA, Bengu WU, Hao ZHENG)
#           2018 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU)
#           2018 Emotech LTD (Author: Xuechen LIU)
# Apache 2.0

#trn_set=/AISHELL-2/iOS/data
#dev_set=/AISHELL-2/iOS/dev
#tst_set=/AISHELL-2/iOS/test
data=data/aishell2
stage=1

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if [ $# != 2 ]; then
  echo "Usage: $0 <corpus-path> <data-path>"
  echo " $0 /export/a05/xna/data/data_aishell data/aishell"
  exit 1;
fi


trn_set=$1/iOS/data
dev_set=$1/iOS/dev
tst_set=$1/iOS/test

train_dir=$data/local/train
dev_dir=$data/local/dev
test_dir=$data/local/test


mkdir -p $train_dir
mkdir -p $dev_dir
mkdir -p $test_dir

echo "**** Creating aishell2 data folder ****"


# download DaCiDian raw resources, convert to Kaldi lexicon format
if [ $stage -le 1 ]; then
  local/prepare_aishell2_dict.sh data/aishell2/local/dict || exit 1;
fi


# wav.scp, text(word-segmented), utt2spk, spk2utt
if [ $stage -le 2 ]; then
  local/prepare_data.sh ${trn_set} data/aishell2/local/dict data/aishell2/local/train data/aishell2/train || exit 1;
  local/prepare_data.sh ${dev_set} data/aishell2/local/dict data/aishell2/local/dev   data/aishell2/dev   || exit 1;
  local/prepare_data.sh ${tst_set} data/aishell2/local/dict data/aishell2/local/test  data/aishell2/test  || exit 1;
fi

mkdir -p $data/train $data/dev $data/test

for f in spk2utt utt2spk wav.scp text; do
  cp $train_dir/$f $data/train/$f || exit 1;
  cp $dev_dir/$f $data/dev/$f || exit 1;
  cp $test_dir/$f $data/test/$f || exit 1;
done

utils/data/validate_data_dir.sh --no-feats $data/train || exit 1;
utils/data/validate_data_dir.sh --no-feats $data/dev || exit 1;
utils/data/validate_data_dir.sh --no-feats $data/test || exit 1;


exit 0;

