#!/bin/bash

# Function to launch experiment
launch_experiment() {
    local fusion=$1
    local num_pre_output_layers=$2
    local output_dir=$3
    local gpu_index=$4

    CUDA_VISIBLE_DEVICES=$gpu_index python main.py --clip_model mobileclip_b --embed_dim $EMBED_DIM --num_pre_output_layers $num_pre_output_layers \
        --num_mapping_layers $NUM_MAPPING_LAYERS --weight_decay 1e-4 --lr 1e-4 --batch_size 64 --epochs 20 \
        --fusion $fusion --clip_grad_norm 0.1 --freeze_clip True --output_dir $output_dir --use_propaganda --use_memotion
}

# Parse command line arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 FUSION GPU_INDEX"
    exit 1
fi

FUSION=$1
GPU_INDEX=$2

# Create the parameters for the experiments
EMBED_DIM=512
NUM_MAPPING_LAYERS=1
NUM_PRE_OUTPUT_LAYERS=(1 3 5 7)

# Loop through configurations
for num_pre_output_layers in ${NUM_PRE_OUTPUT_LAYERS[@]}; do
    output_dir="outputs/${FUSION}_ed${EMBED_DIM}_nml${NUM_MAPPING_LAYERS}_npol${num_pre_output_layers}"
    mkdir -p $output_dir

    launch_experiment $FUSION $num_pre_output_layers $output_dir $GPU_INDEX
    wait
    echo "Finished experiment with fusion $FUSION, num_pre_output_layers $num_pre_output_layers"
done

echo "All experiments finished"
