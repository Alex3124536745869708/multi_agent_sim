#!/bin/bash
#SBATCH --job-name=bla
#SBATCH --output=bla_%j.out
#SBATCH --error=bla_%j.err
#SBATCH --partition=L40s_students
#SBATCH --nodelist=netherite
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

echo "Running on host: $(hostname)"
echo "GPU Verification:"
nvidia-smi

# go to our dir
cd /local

if [ ! -d "$USER" ]; then
	mkdir $USER
fi
cd $USER

# git clone
if [ ! -d "multi_agent_sim" ]; then
	git clone git@github.com:Alex3124536745869708/multi_agent_sim.git
fi
cd multi_agent_sim

# limit threads
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# run experiment
uv run 2_llm_mcp.py

echo "========================================"
echo "All benchmarks finished."
echo "========================================"