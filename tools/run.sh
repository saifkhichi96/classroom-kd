#!/usr/bin/env bash
# --------------------------------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# --------------------------------------------------------------------------------------------------------------------------
# Modified from https://github.com/open-mmlab/mmdetection/blob/3b53fe15d87860c6941f3dda63c0f27422da6266/tools/slurm_train.sh
# --------------------------------------------------------------------------------------------------------------------------

set -x

PARTITION=V100-16GB
JOB_NAME=imagenet_2t0p
GPUS=2
if [ $GPUS -lt 8 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi
CPUS_PER_TASK=${CPUS_PER_TASK:-2}
SRUN_ARGS=${SRUN_ARGS:-""}

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    -K --mem=120G --time=3-00:00 --container-mounts=/netscratch/sarode:/netscratch/sarode,/ds/images:/ds/images --container-image=/netscratch/sarode/imagenet1.sqsh --container-workdir=/netscratch/sarode/Thesis/imagenet-code \
    python our_model.py --config config/mobilenet_2t0p.json