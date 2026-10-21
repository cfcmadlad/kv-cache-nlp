#!/usr/bin/env bash
#SBATCH --account=bits
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --qos=gpu_csis_course
#SBATCH --gres=mps:50
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --open-mode=append

set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source "${KVC_VENV:-$HOME/venvs/kvc}/bin/activate"
exec "$@"
