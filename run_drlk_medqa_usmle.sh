#!/bin/bash

dt=`date '+%Y%m%d_%H%M%S'`
dataset="medqa_usmle"
mode=${1}
# ./LM_models/cambridgeltl--SapBERT-from-PubMedBERT-fulltext
# cambridgeltl/SapBERT-from-PubMedBERT-fulltext
encoder="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
encodername="cambridgeltl--SapBERT-from-PubMedBERT-fulltext"

encoder_lr="5e-5"
decoder_lr="1e-3"
max_seq_len=512
gnn_layer_num=4 # num of gnn layers
max_epochs_before_stop=10
seed=0

# output parameters

echo "***** hyperparameters *****"
echo "dataset: $dataset"
echo "encoder: $encoder"
echo "gnn_layer_num: $gnn_layer_num"
echo "max_seq_len: $max_seq_len"
echo "******************************"

save_dir_name=drlk__ds_${dataset}__enc_${encodername}__gnn_${gnn_layer_num}__sd_${seed}__${dt}
log=logs/${mode}_${dataset}_${save_dir_name}.log.txt

##### Training ######
python3 -u drlk.py --mode $mode \
    --dataset $dataset --encoder $encoder --encoder_lr $encoder_lr --decoder_lr $decoder_lr \
    --max_seq_len $max_seq_len --gnn_layer_num $gnn_layer_num \
    --max_epochs_before_stop $max_epochs_before_stop --seed $seed \
    --save_dir ./saved_models/${dataset}/${save_dir_name}/ --preds_dir ./preds_res/${dataset}/${save_dir_name}/ \
    > ${log} 2>&1 &
echo log: ${log}