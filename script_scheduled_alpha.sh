#!/bin/bash
#SBATCH -p mit_normal_gpu                       # partition name
#SBATCH --job-name=vit_sae_sweep_10-07-25       # name for your job
#SBATCH --gres=gpu:2                            # if you need GPUs
#SBATCH --ntasks=1                              # number of tasks (often 1 for serial jobs)
#SBATCH --cpus-per-task=4                       # CPU cores per task
#SBATCH --mem=79GB                                # memory per node
#SBATCH --time=06:00:00                         # max walltime (HH:MM:SS)
#SBATCH --output=./slurm_logs/slurm-%j.out      # output file (%j = job ID) to capture logs for debugging
#SBATCH --mail-type=FAIL                        # when to send emails
#SBATCH --mail-user=hann.yun.gao@gmail.com      # email address

# Load your shell environment to activate your Conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate interp_robustness
which python

echo "Python path: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

epochs=30
lr=0.001
batch_size=64
alpha_list=(
    -1
)

no_freeze=10

beta_list=(
    # 0.5
    1.0
    # 2.0
)

expansion_factor_list=(
    # 2
    # 4
    32
)

target_module="vit.encoder.layer.11.output" # dim 3072

project="6.7960_project_FIXED_$epochs=${epochs}_lr=${lr}_no_freeze=${no_freeze}"

for alpha in "${alpha_list[@]}"; do
    for beta in "${beta_list[@]}"; do
         for expansion in "${expansion_factor_list[@]}"; do
            python main.py \
                --project "$project" \
                --epochs $epochs \
                --lr $lr \
                --batch_size $batch_size \
                --alpha $alpha \
                --beta $beta \
                --no_freeze $no_freeze \
                --expansion_factor $expansion \
                --target_module "$target_module"
        done
    done
done