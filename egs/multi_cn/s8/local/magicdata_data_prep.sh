#!/bin/bash

# Copyright 2019 Xingyu Na
# Apache 2.0

. ./path.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 <corpus-path> <data-path>"
  echo " $0 /export/a05/xna/data/magicdata data/magicdata"
  exit 1;
fi

corpus=$1
data=$2

if [ ! -d $corpus/train ] || [ ! -d $corpus/dev ] || [ ! -d $corpus/test ]; then
  echo "Error: $0 requires complete corpus"
  exit 1;
fi

echo "**** Creating magicdata data folder ****"

mkdir -p $data/{train,dev,test,tmp}

# find wav audio file for train, dev and test resp.
tmp_dir=$data/tmp
find $corpus -iname "*.wav" > $tmp_dir/wav.flist
n=`cat $tmp_dir/wav.flist | wc -l`
[ $n -ne 609552 ] && \
  echo Warning: expected 609552 data data files, found $n



local/prepare_aishell2_dict.sh $2/local/dict || exit 1;
dict_dir=$2/local/dict



for x in train dev test; do
  grep -i "/$x/" $tmp_dir/wav.flist > $data/$x/wav.flist || exit 1;
  echo "Filtering data using found wav list and provided transcript for $x"
  local/magicdata_data_filter.py $data/$x/wav.flist $corpus/$x/TRANS.txt $data/$x local/magicdata_badlist
  
  python -c "import jieba" 2>/dev/null || \
  (echo "jieba is not found. Use tools/extra/install_jieba.sh to install it." && exit 1;)

  cat $data/$x/transcripts.txt |\
    sed 's/！//g' | sed 's/？//g' |\
    sed 's/，//g' | sed 's/－//g' |\
    sed 's/：//g' | sed 's/；//g' |\
    sed 's/　//g' | sed 's/。//g' |\
    sed 's/\[//g' | sed 's/\]//g' |\
    sed 's/\《//g' | sed 's/\》//g' |\
    sed 's/\……/SPK/g' | sed 's/"//g' |\
    sed 's/\\//g' | sed 's/、//g' |\
    sed 's/,//g' | sed 's/?//g'|\
    sort -k 1 | uniq > $data/$x/trans.txt
    perl -p -e 's/\r//g' $data/$x/trans.txt > $data/$x/trans_1.txt

    awk '{print $1}' $dict_dir/lexicon.txt | sort | uniq | awk '{print $1,99}'> $data/$x/word_seg_vocab.txt
    python local/word_segmentation.py $data/$x/word_seg_vocab.txt $data/$x/trans_1.txt| tr '[a-z]' '[A-Z]' | sed 's/FIL/[FIL]/g' | sed 's/SPK/[SPK]/'| awk '{if (NF > 1) print $0;}'  > $data/$x/text

  for file in wav.scp utt2spk text; do
    sort $data/$x/$file -o $data/$x/$file
  done
  utils/utt2spk_to_spk2utt.pl $data/$x/utt2spk > $data/$x/spk2utt
done

rm -r $tmp_dir

utils/data/validate_data_dir.sh --no-feats $data/train || exit 1;
utils/data/validate_data_dir.sh --no-feats $data/dev || exit 1;
utils/data/validate_data_dir.sh --no-feats $data/test || exit 1;
