#!usr/bin/env python3.9

data=./result/audios
stage=1


ngpu=0
dumpdir=data

# TasNet Config
sample_rate=8000
# Network config
L=40
N=500
hidden_size=200
num_layers=4
bidirectional=1
nspk=2
e_type=conv
# Training config
use_cuda=1
epochs=50
shuffle=0
half_lr=0
early_stop=0
max_norm=5
# minibatch
batch_size=10
num_workers=2
# optimizer
optimizer=adam
lr=1e-3
momentum=0
l2=1e-5

tag="" 
expdir=./checkpoints/${tag}/
if [ $stage -le 0 ]; then
  echo "Stage 0: Prepare mixed audios"
  ./data_prepare/run.sh
fi

if [ $stage -le 1 ]; then
  echo "Stage 1: Generating json files including wav path and duration"
  [ ! -d $dumpdir ] && mkdir $dumpdir
  python3 preprocess.py --in-dir $data --out-dir $dumpdir --sample-rate $sample_rate
fi


if [ $stage -le 2 ]; then
  echo "Stage 2: Training"
    
    python3  train.py \
    --use_cuda $use_cuda \
    --train_dir $dumpdir/tr \
    --valid_dir $dumpdir/cv \
    --sample_rate $sample_rate \
    --L $L \
    --N $N \
    --hidden_size $hidden_size \
    --num_layers $num_layers \
    --bidirectional $bidirectional \
    --nspk $nspk \
    --epochs $epochs \
    --half_lr $half_lr \
    --early_stop $early_stop \
    --max_norm $max_norm \
    --shuffle $shuffle \
    --batch_size $batch_size \
    --num_workers $num_workers \
    --optimizer $optimizer \
    --lr $lr \
    --momentum $momentum \
    --l2 $l2 \
    --save_folder ${expdir} \
    --e_type $e_type
fi


if [ $stage -le 3 ]; then
  echo "Stage 3: Evaluate separation performance"
  python3 eval.py \
    --model_path ${expdir}/final.pth.tar \
    --data_dir $dumpdir/tt \
    --cal_sdr 1 \
    --use_cuda $use_cuda \
    --sample_rate $sample_rate \
    --batch_size 10
fi


if [ $stage -le 4 ]; then
  echo "Stage 4: Separate speech using TasNet"
  separate_dir=${expdir}/separate
  python3.9 separate.py \
    --model_path ${expdir}/final.pth.tar \
    --mix_json $dumpdir/tt/mix.json \
    --out_dir ${separate_dir} \
    --use_cuda $use_cuda \
    --sample_rate $sample_rate \
    --batch_size 10
fi
