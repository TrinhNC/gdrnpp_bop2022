#!/usr/bin/env bash
set -x
this_dir=$(dirname "$0")
# commonly used opts:

#resume or pretrained, or test checkpoint
# need to remove the solver states from the checkpoint using /tools/remove_optim_from_ckpt.py before put it here
MODEL.WEIGHTS="/home/robodev/Documents/BPC/gdrnpp_bop2022/output/gdrn/ipdPbrSO/4/model_0044429_wo_optim.pth"
CFG=$1
CUDA_VISIBLE_DEVICES=$2
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
# GPUS=($(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n'))
NGPU=${#GPUS[@]}  # echo "${GPUS[0]}"
echo "use gpu ids: $CUDA_VISIBLE_DEVICES num gpus: $NGPU"
# CUDA_LAUNCH_BLOCKING=1
NCCL_DEBUG=INFO
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
PYTHONPATH="$this_dir/../..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$2 python3 $this_dir/main_gdrn.py \
    --config-file $CFG --num-gpus $NGPU  ${@:3}
