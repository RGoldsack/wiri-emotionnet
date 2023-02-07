#!/bin/bash -e
#
#SBATCH --job-name=LSTMprep
#SBATCH --partition=gpu
#SBATCH -o LSTMprep.out
#SBATCH -e LSTMprep.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --mem=1G
#

module load GCCcore/11.3.0
module load Python/3.10.4

python LSTMsubmit.py 8

# dos2unix LSTMsubmit.sh

sbatch LSTMsubmit.sh


