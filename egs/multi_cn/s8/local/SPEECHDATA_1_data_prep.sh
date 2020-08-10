. ./path.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 <corpus-path> <data-path>"
  echo " $0 /export/a05/xna/data/fayuan data/fayuan"
  exit 1;
fi


audio_dir=$1/wav
#text_dir=$1/fayuan.txt
text_dir=$1/trans.txt
#data=data/baseline_1
data=$2


train_dir=$data/local/train
#dev_dir=$data/local/dev
#test_dir=$data/local/test
tmp_dir=$data/local/tmp
#dir=$test_dir

mkdir -p $train_dir
#mkdir -p $dev_dir
#mkdir -p $test_dir
mkdir -p $tmp_dir


# data directory check
if [ ! -d $audio_dir ] || [ ! -f $text_dir ]; then
  echo "Error: $0 requires two directory arguments"
  exit 1;
fi

echo "**** Creating fayuan data folder ****"


# find wav audio file for train, dev and test resp.
find $audio_dir -iname "*.wav" > $tmp_dir/wav.flist
n=`cat $tmp_dir/wav.flist | wc -l`
echo $n
# 待修改具体数目
[ $n -ne 34294 ] && \
  echo Warning: expected 34294 data data files, found $n

grep -i "wav/train" $tmp_dir/wav.flist > $train_dir/wav.flist || exit 1;
#grep -i "wav/dev" $tmp_dir/wav.flist > $dev_dir/wav.flist || exit 1;
#grep -i "wav/test" $tmp_dir/wav.flist > $test_dir/wav.flist || exit 1;

rm -r $tmp_dir

# download DaCiDian raw resources, convert to Kaldi lexicon format
local/prepare_aishell2_dict.sh $2/local/dict || exit 1;

dict_dir=$2/local/dict

# Transcriptions preparation
for dir in $train_dir; do
   echo Preparing $dir transcriptions
   sed -e 's/\.wav//' $dir/wav.flist | awk -F '/' '{print $NF}' > $dir/utt.list
   sed -e 's/\.wav//' $dir/wav.flist | awk -F '/' '{i=NF-1;printf("%s %s\n",$NF,$i)}' > $dir/utt2spk_all
   paste -d' ' $dir/utt.list $dir/wav.flist > $dir/wav.scp_all
   
   #TEXT
   python -c "import jieba" 2>/dev/null || \
  (echo "jieba is not found. Use tools/extra/install_jieba.sh to install it." && exit 1;)

   utils/filter_scp.pl -f 1 $dir/utt.list $text_dir | \
     sed 's/，//g' | sed 's/。//g' | sed 's/？//g' | sed 's/、//g' | sed 's/！//g' | sed 's/：//g' |\
     sort -k 1 | uniq > $dir/trans.txt
     perl -p -e 's/\r//g' $dir/trans.txt > $dir/trans_1.txt
     awk '{print $1}' $dict_dir/lexicon.txt | sort | uniq | awk '{print $1,99}'> $dir/word_seg_vocab.txt
     python local/word_segmentation.py $dir/word_seg_vocab.txt $dir/trans_1.txt > $dir/text
  

   utils/filter_scp.pl -f 1 $dir/utt.list $dir/utt2spk_all | sort -n | awk '{print $1" "$2}' > $dir/utt2spk
   utils/filter_scp.pl -f 1 $dir/utt.list $dir/wav.scp_all | sort -n > $dir/wav.scp
   
   for file in wav.scp utt2spk text; do
    sort $dir/$file -o $dir/$file
   done

   utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt
done

mkdir -p $data/train

for f in spk2utt utt2spk wav.scp text; do
  cp $train_dir/$f $data/train/$f || exit 1;
  #cp $dev_dir/$f $data/dev/$f || exit 1;
  #cp $test_dir/$f $data/test/$f || exit 1;
done

utils/data/validate_data_dir.sh --no-feats $data/train || exit 1;
#utils/data/validate_data_dir.sh --no-feats $data/dev || exit 1;
#utils/data/validate_data_dir.sh --no-feats $data/test || exit 1;

echo "$0: fayuan data preparation succeeded"
exit 0;





