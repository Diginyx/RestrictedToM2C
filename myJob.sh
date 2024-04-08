#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=ToM2C
#SBATCH --partition=standard
#SBATCH --time=00-08:00:00
# --gpus=1
# --cpus-per-gpu=4
#SBATCH --mem-per-cpu=32GB
#SBATCH --account=chaijy0

# The application(s) to execute along with its input arguments and options:

module load python/3.9.12
source venv/bin/activate
python main.py --env CN --model ToM2C --workers 6 --env-steps 10 --A2C-steps 10 --norm-reward
