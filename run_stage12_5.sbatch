#!/usr/bin/env bash
#SBATCH --job-name=synvqa12_5
#SBATCH --account=bgek-delta-cpu
#SBATCH --partition=cpu
#SBATCH --output=/u/yhuang48/Memory2Param/slurm-%j.out
#SBATCH --error=/u/yhuang48/Memory2Param/slurm-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

cd /u/yhuang48/Memory2Param/synvqa

python -m synvqa.run_pipeline \
  --config configs/pipeline_stage12_5.yaml \
  --seeds seeds/anchor_seeds_5.jsonl \
  --output data/stage12_5_smoke2 \
  --stages 1,2
