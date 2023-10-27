#!usr/bin/env python3.9


data=./audios
stage=1
# --
expdir=./checkpoints
ngpu=0
dumpdir=data

# -- START TasNet Config
sample_rate=8000
# Network config
L=40
N=500
hidden_size=200
num_layers=4
bidirectional=1
nspk=2
# Training config
use_cuda=0
epochs=5
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
# save and visualize
checkpoint=0
continue_from=""
print_freq=10
visdom=0
visdom_epoch=0
visdom_id="TasNet Training"
# -- END TasNet Config

# exp tag
tag="" # tag for managing experiments.


if [ $stage -le 1 ]; then
  echo "Stage 1: Generating json files including wav path and duration"
  [ ! -d $dumpdir ] && mkdir $dumpdir
  python3.9 preprocess.py --in-dir $data --out-dir $dumpdir --sample-rate $sample_rate
fi


if [ -z ${tag} ]; then
  log=log/train_r${sample_rate}_L${L}_N${N}_h${hidden_size}_l${num_layers}_bi${bidirectional}_C${nspk}_epoch${epochs}_half${half_lr}_norm${max_norm}_bs${batch_size}_worker${num_workers}_${optimizer}_lr${lr}_mmt${momentum}_l2${l2}
else
  log=log/train_${tag}
fi

if [ $stage -le 2 ]; then
  echo "Stage 2: Training"
  # ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
    python3.9  train.py \
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
    --checkpoint $checkpoint \
    --continue_from "$continue_from" \
    --print_freq ${print_freq} \
    --visdom $visdom \
    --visdom_epoch $visdom_epoch \
    --visdom_id "$visdom_id"
fi


if [ $stage -le 3 ]; then
  echo "Stage 3: Evaluate separation performance"
  python3.9 eval.py \
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
