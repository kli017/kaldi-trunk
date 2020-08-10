#!/bin/bash

# This script is copied from librispeech/s5
# In a previous version, pitch is used with hires mfcc, however,
# removing pitch does not cause regression, and helps online
# decoding, so pitch is removed in this recipe.

# This is based on tdnn_1d_sp, but adding cnn as the front-end.
# The cnn-tdnn-f (tdnn_cnn_1a_sp) outperforms the tdnn-f (tdnn_1d_sp).

# local/chain/compare_cer.sh --online exp/chain_cleaned/tdnn_cnn_1a_pitch_sp exp/chain_nopitch/tdnn_cnn_1a_sp
# System                      tdnn_cnn_1a_pitch_sp tdnn_cnn_1a_sp
# CER on aidatatang(tg)            4.99      4.98
#             [online:]          4.99      4.98
# CER on aishell(tg)               6.01      5.90
#             [online:]          6.01      5.90
# CER on magicdata(tg)             4.21      4.24
#             [online:]          4.23      4.25
# CER on thchs30(tg)              13.02     12.96
#             [online:]         13.00     12.94
# Final train prob              -0.0436   -0.0438
# Final valid prob              -0.0553   -0.0544
# Final train prob (xent)       -0.8083   -0.8157
# Final valid prob (xent)       -0.8766   -0.8730
# Num-parameters               19141072  19141072

set -e

# configs for 'chain'
stage=0
decode_nj=10
train_set=train_all_cleaned
gmm=tri4a_cleaned
nnet3_affix=_cleaned_3

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
affix=cnn_1a_3
tree_affix=
train_stage=-10
get_egs_stage=-10

# TDNN options
frames_per_eg=150,110,100
remove_egs=true
common_egs_dir=
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'

#test_sets="aishell aidatatang magicdata thchs"
test_sets=""
test_online_decoding=true  # if true, it will run the last decoding stage.
#pre_train=exp/chain${nnet3_affix}/tdnn_cnn_1a_3_sp/539.mdl

# configs for transfer learning
src_mdl=exp/chain/tdnn1d_sp/final.mdl # Input chain model
                                                   # trained on source dataset (wsj).
                                                   # This model is transfered to the target domain.

src_mfcc_config=conf/mfcc_hires.conf # mfcc config used to extract higher dim
                                                  # mfcc features for ivector and DNN training
                                                  # in the source domain.
src_ivec_extractor_dir=exp/nnet3_cleaned_3/extractor  # Source ivector extractor dir used to extract ivector for
                         # source data. The ivector for target data is extracted using this extractor.
                         # It should be nonempty, if ivector is used in the source model training.
src_lang=../../wsj/s5/data/lang # Source lang directory used to train source model.
                                # new lang dir for transfer learning experiment is prepared
                                # using source phone set phones.txt and lexicon.txt
                                # in src lang and dict dirs and words.txt in target lang dir.

src_dict=../../wsj/s5/data/local/dict_nosp  # dictionary for source dataset containing lexicon.txt,
                                            # nonsilence_phones.txt,...
                                            # lexicon.txt used to generate lexicon.txt for
                                            # src-to-tgt transfer.

src_gmm_dir=../../wsj/s5/exp/tri4b # source gmm dir used to generate alignments
                                   # for target data.

src_tree_dir=../../wsj/s5/exp/chain/tree_a_sp # chain tree-dir for src data;
                                         # the alignment in target domain is
                                         # converted using src-tree




common_egs_dir=
primary_lr_factor=0.25 # The learning-rate factor for transferred layers from source
                       # model. e.g. if 0, the paramters transferred from source model
                       # are fixed.
                       # The learning-rate factor for new added layers is 1.0.
phone_lm_scales="1,10" # comma-separated list of positive integer multiplicities
                       # to apply to the different source data directories (used
                       # to give the RM data a higher weight).

nnet_affix=_1epoch_transfer


# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

required_files="$src_mfcc_config $src_mdl $src_lang/phones.txt $src_dict/lexicon.txt $src_gmm_dir/final.mdl $src_tree_dir/tree"


use_ivector=false
ivector_dim=$(nnet3-am-info --print-args=false $src_mdl | grep "ivector-dim" | cut -d" " -f2)
if [ -z $ivector_dim ]; then ivector_dim=0 ; fi

if [ ! -z $src_ivec_extractor_dir ]; then
  if [ $ivector_dim -eq 0 ]; then
    echo "$0: Source ivector extractor dir '$src_ivec_extractor_dir' is specified "
    echo "but ivector is not used in training the source model '$src_mdl'."
  else
    required_files="$required_files $src_ivec_extractor_dir/final.dubm $src_ivec_extractor_dir/final.mat $src_ivec_extractor_dir/final.ie"
    use_ivector=true
  fi
else
  if [ $ivector_dim -gt 0 ]; then
    echo "$0: ivector is used in training the source model '$src_mdl' but no "
    echo " --src-ivec-extractor-dir option as ivector dir for source model is specified." && exit 1;
  fi
fi


for f in $required_files; do
  if [ ! -f $f ]; then
    echo "$0: no such file $f" && exit 1;
  fi
done

#if [ $stage -le -1 ]; then
#  echo "$0: Prepare lang for RM-WSJ using WSJ phone set and lexicon and RM word list."
#  if ! cmp -s <(grep -v "^#" $src_lang/phones.txt) <(grep -v "^#" data/lang/phones.txt); then
#    local/prepare_wsj_rm_lang.sh  $src_dict $src_lang $lang_src_tgt
#  else
#    rm -rf $lang_src_tgt 2>/dev/null || true
#    cp -r data/lang $lang_src_tgt
#  fi
#fi




# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 11" if you have already
# run those things.
local/chain/run_ivector_common.sh --stage $stage \
                                  --train-set $train_set \
                                  --test-sets "$test_sets" \
                                  --gmm $gmm \
                                  --nnet3-affix "$nnet3_affix" || exit 1;

gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali_${train_set}_sp
tree_dir=exp/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}
lang=data/lang_chain
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=exp/chain${nnet3_affix}/tdnn${affix:+_$affix}_sp_new
train_data_dir=data/${train_set}_sp_hires
lores_train_data_dir=data/${train_set}_sp
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires

# if we are using the speed-perturbed data we need to generate
# alignments for it.

for f in $gmm_dir/final.mdl $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
    $lores_train_data_dir/feats.scp $ali_dir/ali.1.gz; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

# Please take this as a reference on how to specify all the options of
# local/chain/run_chain_common.sh
local/chain/run_chain_common.sh --stage $stage \
                                --gmm-dir $gmm_dir \
                                --ali-dir $ali_dir \
                                --lores-train-data-dir ${lores_train_data_dir} \
                                --lang $lang \
                                --lat-dir $lat_dir \
                                --num-leaves 9000 \
                                --tree-dir $tree_dir || exit 1;

#if [ $stage -le 14 ]; then
#  echo "$0: creating neural net configs using the xconfig parser";
#  num_targets=$(tree-info $tree_dir/tree | grep num-pdfs | awk '{print $2}')
#  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
#  cnn_opts="l2-regularize=0.01"
#  ivector_affine_opts="l2-regularize=0.0"
#  affine_opts="l2-regularize=0.008 dropout-proportion=0.0 dropout-per-dim=true dropout-per-dim-continuous=true"
#  tdnnf_first_opts="l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.0"
#  tdnnf_opts="l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.75"
#  linear_opts="l2-regularize=0.008 orthonormal-constraint=-1.0"
#  prefinal_opts="l2-regularize=0.008"
#  output_opts="l2-regularize=0.005"
#
#  mkdir -p $dir/configs
#
#  cat <<EOF > $dir/configs/network.xconfig
#  tdnnf-layer name=tfnnf-target $tdnnf_opts input=Append(idct-batchnorm, ivector-batchnorm) dim=1536 bottleneck-dim=160 time-stride=3
#  tdnnf-layer name=tdnnf18 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
#  linear-component name=prefinal-l dim=256 $linear_opts
#
#  ## adding the layers for chain branch
#  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=1536 small-dim=256
#  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts
#
#  # adding the layers for xent branch
#  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=1536 small-dim=256
#  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
#EOF
#  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
#fi

if [ $stage -le 13 ]; then
  # Set the learning-rate-factor for all transferred layers but the last output
  # layer to primary_lr_factor.
  $train_cmd $dir/log/generate_input_mdl.log \
    nnet3-am-copy --raw=true --edits="set-learning-rate-factor name=* learning-rate-factor=$primary_lr_factor; set-learning-rate-factor name=output* learning-rate-factor=1.0" \
      $src_mdl $dir/input.raw || exit 1;
fi

if [ $stage -le 14 ]; then
  echo "$0: compute {den,normalization}.fst using weighted phone LM with wsj and rm weight $phone_lm_scales."
  steps/nnet3/chain/make_weighted_den_fst.sh --cmd "$train_cmd" \
    --num-repeats $phone_lm_scales \
    --lm-opts '--num-extra-lm-states=200' \
    $src_tree_dir $lat_dir $dir || exit 1;
fi



if [ $stage -le 15 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/multi_cn-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/chain/train.py --stage $train_stage \
    --use-gpu "wait" \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0 --constrained false" \
    --egs.chunk-width $frames_per_eg \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --trainer.num-chunk-per-minibatch 128,64 \
    --trainer.frames-per-iter 3000000 \
    --trainer.num-epochs 1 \
    --trainer.input-model \
    --trainer.optimization.num-jobs-initial 4 \
    --trainer.optimization.num-jobs-final 4 \
    --trainer.optimization.initial-effective-lrate 0.00015 \
    --trainer.optimization.final-effective-lrate 0.000015 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir  || exit 1;

fi

graph_dir=$dir/graph_tg
if [ $stage -le 16 ]; then
  # Note: it's not important to give mkgraph.sh the lang directory with the
  # matched topology (since it gets the topology file from the model).
  utils/mkgraph.sh --self-loop-scale 1.0 --remove-oov \
    data/lang_combined_tg $dir $graph_dir
  # remove <UNK> (word id is 3) from the graph, and convert back to const-FST.
  fstrmsymbols --apply-to-output=true --remove-arcs=true "echo 3|" $graph_dir/HCLG.fst - | \
    fstconvert --fst_type=const > $graph_dir/temp.fst
  mv $graph_dir/temp.fst $graph_dir/HCLG.fst
fi

if [ $stage -le 17 ]; then
  rm $dir/.error 2>/dev/null || true
  for decode_set in $test_sets; do
    (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
        --nj $decode_nj --cmd "$decode_cmd" \
        --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${decode_set}_hires \
        $graph_dir data/${decode_set}/test_hires $dir/decode_${decode_set}_tg || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

if $test_online_decoding && [ $stage -le 18 ]; then
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
  steps/online/nnet3/prepare_online_decoding.sh \
    --mfcc-config conf/mfcc_hires.conf \
    $lang exp/nnet3${nnet3_affix}/extractor $dir ${dir}_online

  rm $dir/.error 2>/dev/null || true
  for data in $test_sets; do
    (
      nspk=$(wc -l <data/${data}/test/spk2utt)
      # note: we just give it "data/${data}" as it only uses the wav.scp, the
      # feature type does not matter.
      steps/online/nnet3/decode.sh \
        --acwt 1.0 --post-decode-acwt 10.0 \
        --nj $nspk --cmd "$decode_cmd" \
        $graph_dir data/${data}/test ${dir}_online/decode_${data}_tg || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

exit 0;
