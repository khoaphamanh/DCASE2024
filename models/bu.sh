#!/bin/bash

#SBATCH --mail-user=phamanh@tnt.uni-hannover.de # only <UserName>@tnt.uni-hannover.de is allowed as mail address
#SBATCH --mail-type=ALL             # Eine Mail wird bei Job-Start/Ende versendet
#SBATCH --job-name=bu    # Name unter dem der Job in der Job-History gespeichert wird
#SBATCH --output=/home/phamanh/nobackup/DCASE2024/models/result/bu-%j.txt   # Logdatei für den merged STDOUT/STDERR output (%j wird durch slurm job-ID ersetzt)

#SBATCH --time=24:00:00             # Maximale Laufzeit des Jobs, bis Slurm diesen abbricht (HH:MM:SS)
#SBATCH --partition=gpu_normal_stud   # Partition auf der gerechnet werden soll. Ohne Angabe des Parameters wird auf der
                                    #   Default-Partition gerechnet. Es können mehrere angegeben werden, mit Komma getrennt.
#SBATCH --cpus-per-task=4          # Reservierung von 4 CPUs pro Rechenknoten
#SBATCH --mem=64G                   # Reservierung von 10GB RAM
#SBATCH --gres=gpu:1
# #SBATCH --exclude=cc1g07  # dont use ndoe cc1g07 

source /home/phamanh/anaconda3/bin/activate /home/phamanh/anaconda3/envs/dcase
cd /home/phamanh/nobackup/DCASE2024/models

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "CPU cores: ${SLURM_JOB_CPUS_PER_NODE}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Memory: ${SLURM_MEM_PER_NODE}" 
echo "Python environment: $(conda info --envs | grep '*' | sed -e 's/^[ \t*]*//')"
echo "current work directory: ${PWD}"
echo 

python /home/phamanh/nobackup/DCASE2024/models/batch_uniform.py