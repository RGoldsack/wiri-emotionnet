#!/usr/bin/env python
# coding: utf-8

import json 

params = {}
i = 1

for split in range(0, 8):
    for emo in ["cont"]: #["cont", "6emo", "sum.PANAS"]:
        for phys in ["phys", "mocap", "both"]:
            for rand in ["rand", "observed"]:
                for valence in ["both", "pos", "neg"]:
                    if emo == "cont":
                        for freq in ["66.66L", "1S", "5S", "10S"]:
                            name = "LSTM" + str(split) + "_" + str(emo)[:1].upper() + "_" + str(phys)[:1].upper() + "_" + str(rand)[:1].upper() + "_V" + str(valence)[:1].upper() + "_" + freq[0:2]
                            params[i] = [name, split, emo, phys, rand, valence, freq]
                            i += 1
                    else:
                        name = "LSTM" + str(split) + "_" + str(emo)[:1].upper() + "_" + str(phys)[:1].upper() + "_" + str(rand)[:1].upper() + "_V" + str(valence)[:1].upper()
                        params[i] = [name, split, emo, phys, rand, valence, ""]
                        i += 1
                
with open("params.txt", "w") as fp:
    json.dump(params, fp) 
                        
n_jobs = len(params)

myBat = open("LSTMsubmit.sh", "wt")

text = "#!/bin/bash\n#\n#SBATCH -a 1-" + str(n_jobs) + "\n#SBATCH -o errout2/LSTMjob_%a.out\n#SBATCH -e errout2/LSTMjob_%a.err\n#SBATCH --time=1-00:00:00\n#SBATCH --partition=gpu\n#SBATCH --gres=gpu:1\n#SBATCH --ntasks=1\n#SBATCH --mem=150G\n#\nmodule load singularity/3.7.3\nmodule load CUDA/11.3.1\nmodule load GCC/10.3.0\nmodule load OpenMPI/4.1.1\nmodule load TensorFlow/2.6.0-CUDA-11.3.1\nmodule load cuDNN/8.2.1.32-CUDA-11.3.1\n#\n#run the container with the runscript defined when we created it\n#nvidia-smi\n\npython LSTM_MO_AutoCV.py ${SLURM_ARRAY_TASK_ID}"

myBat.write(text)
myBat.close()


