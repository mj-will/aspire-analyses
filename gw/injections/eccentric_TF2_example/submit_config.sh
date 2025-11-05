#!/bin/bash
#
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=eccentric_PE
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/eccentric_PE_%j.out
#SBATCH --error=logs/eccentric_PE_%j.err
#SBATCH --partition=sciama3.q,sciama3-5.q,sciama4.q,sciama5-5.q

module purge
module load system anaconda3

ENV=/mnt/lustre2/shared_conda/envs/mjwill/aspire/

echo "Activating conda environment..."
source activate ${ENV}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Print the python path
echo "Python path: $(which python)"
# Run the Python script with the number of tasks per node
# ${ENV}/bin/python run_from_config.py --n-pool ${SLURM_NTASKS_PER_NODE} --sampler-config dynesty.json --seed 1234 --outdir outdir_from_config --label dynesty_aligned

# ${ENV}/bin/python run_from_config.py --n-pool ${SLURM_NTASKS_PER_NODE} --sampler-config dynesty.json --seed 1234 --outdir outdir_from_config --label dynesty_eccentric --eccentric

${ENV}/bin/python run_from_config.py --n-pool ${SLURM_NTASKS_PER_NODE} --sampler-config aspire.json --seed 1234 --outdir outdir_from_config --label aspire_from_aligned_Exp5 --eccentric
