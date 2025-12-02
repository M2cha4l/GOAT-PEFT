# set it 
BASE_DIR=/Users/m2cha4l/Downloads/GOAT-PEFT #e.g. /home/xxx/GOAT-PEFT
OUT_DIR=/Users/m2cha4l/Downloads/GOAT-PEFT/results/glue #e.g. /mnt/models/
cd $BASE_DIR

# MacBook 
CUDA_NUM=1
run_command="python"

set -xe

CONDA_INIT=false
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    CONDA_INIT=true
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    CONDA_INIT=true
elif [ -f "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]; then
    source "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
    CONDA_INIT=true
fi

# 激活 conda 环境
if [ "$CONDA_INIT" = true ]; then
    conda activate goat
else
    # 如果 conda 未初始化，尝试直接使用 conda 环境的 python
    if [ -d "$HOME/miniconda3/envs/goat" ]; then
        export PATH="$HOME/miniconda3/envs/goat/bin:$PATH"
        echo "Using conda environment from: $HOME/miniconda3/envs/goat"
    elif [ -d "$HOME/anaconda3/envs/goat" ]; then
        export PATH="$HOME/anaconda3/envs/goat/bin:$PATH"
        echo "Using conda environment from: $HOME/anaconda3/envs/goat"
    else
        echo "Warning: Could not find conda environment 'goat'. Using system python."
        echo "If you have conda installed, run: conda init zsh"
    fi
fi

cd $BASE_DIR/goat

MOE() {
export ETA=1.0
lora=src.goat
# MacBook memory limited, reduce batch size
totalbz=32
model=roberta-large
# rank=8
# alpha=16
rank=32
alpha=64
# MacBook single gpu, reduce batch size
bz=${bz:-8}
gacc=$(( totalbz / bz / CUDA_NUM ))
ep=1
lr=1e-4
k=${k:-2}
e=8
aux=1e-3

unset WANDB_MODE
if [ -n "$DEBUG" ]; then
  export WANDB_MODE=disabled
fi

if [[ $lora == *"ft"* ]]; then
    lr=1e-5 
elif [[ "$task" == *"rte"* ]]; then
    lr=2e-5
else
    lr=1e-4
fi
# mrpc rte cola sst2 qnli mnli qqp
for task in mrpc; do
    if [[ "$lora" == *"moe"* ]]; then
        prj=$model-$task-${lora}a${aux}-${k}in${e}-total${totalbz}dp${CUDA_NUM}bz${bz}lr${lr}
    else
        prj=$model-$task-${lora}-total${totalbz}dp${CUDA_NUM}bz${bz}lr${lr}
    fi

    if [[ "$lora" == *"lora"* ]]; then
        prj+="r${rank}a${alpha}"
    fi

    if [[ "$task" == *"rte"* ]]; then
        ep=50
    else
        ep=10
    fi
    out="$OUT_DIR/$prj"

    $run_command train_nlu.py \
        --lora $lora \
        --task glue-mlm-$task \
        --bz $bz \
        --model $model \
        --gacc $gacc \
        --ep $ep \
        --aux_loss_coeff=$aux \
        --experts=$e \
        --k $k \
        --lr $lr \
        --prj $prj \
        --rank $rank \
        --alpha $alpha \
        --output $out \
        --seed 0 \
        --result $BASE_DIR/goat/results/glue \
        --git_hash $(git rev-parse --short HEAD 2>/dev/null || echo "unknown") 

    lora_dirs+=($prj)
done

}

MOE
