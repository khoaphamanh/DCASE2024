#!/bin/bash

#SBATCH --mail-user=phamanh@tnt.uni-hannover.de # only <UserName>@tnt.uni-hannover.de is allowed as mail address
#SBATCH --mail-type=ALL             # Eine Mail wird bei Job-Start/Ende versendet
#SBATCH --job-name=w2v_demo      # Name unter dem der Job in der Job-History gespeichert wird
#SBATCH --output=/home/phamanh/nobackup/DCASE2024/models/result/w2v-%j.txt   # Logdatei für den merged STDOUT/STDERR output (%j wird durch slurm job-ID ersetzt)

#SBATCH --time=24:00:00             # Maximale Laufzeit des Jobs, bis Slurm diesen abbricht (HH:MM:SS)
#SBATCH --partition=gpu_normal_stud   # Partition auf der gerechnet werden soll. Ohne Angabe des Parameters wird auf der
                                    #   Default-Partition gerechnet. Es können mehrere angegeben werden, mit Komma getrennt.
#SBATCH --cpus-per-task=8          # Reservierung von 4 CPUs pro Rechenknoten
#SBATCH --mem=64G                   # Reservierung von 10GB RAM
#SBATCH --gres=gpu:4 

source /home/phamanh/anaconda3/bin/activate /home/phamanh/anaconda3/envs/dcase
cd /home/phamanh/nobackup/DCASE2024/models

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "CPU cores: ${SLURM_JOB_CPUS_PER_NODE}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Python environment: $(conda info --envs | grep '*' | sed -e 's/^[ \t*]*//')"
echo 

python3 /home/phamanh/nobackup/DCASE2024/models/w2v.py