#!/usr/bin/env python
# coding: utf-8

import glob
import sys
from shutil import rmtree
from os import mkdir, getcwd
from os.path import isdir

# numSplits = int(sys.argv[1])
numSplits = 4
job_list = []
gpu_type = {"phys" : "P100", "mocap": "A100", "both": "A100"}
location = "LSTM/"
if isdir(location):
    rmtree(location)
    mkdir(location[:-1])
else:
    mkdir(location[:-1])

if (getcwd() == "/nfs/home/goldsaro"): # Raapoi
    for split in range(0, 8):
        for emo in ["cont"]: #["cont", "6emo", "sum.PANAS"]:
            for phys in ["phys", "mocap", "both"]:
                for rand in ["shuf", "rand", "observed"]:
                    for valence in ["both", "pos", "neg"]:
                        name = "LSTM" + str(split) + "_" + str(emo)[:1].upper() + "_" + str(phys)[:1].upper() + "_" + str(rand)[:1].upper() + "_V" + str(valence)[:1].upper()
                        filename = location + name
                        myBat = open('%s.sh' % filename, 'wt')
                        text = "#!/bin/bash\n#\n#SBATCH --job-name=" + name + "\n" + "#SBATCH -o errout/" + name + ".out" + "\n" + "#SBATCH -e errout/" + name + ".err" + "\n" + "#SBATCH --time=1-00:00:00\n#SBATCH --partition=gpu\n#SBATCH --gres=gpu:1\n#SBATCH --ntasks=1\n#SBATCH --mem=150G\n#\nmodule load singularity/3.7.3\nmodule load CUDA/11.3.1\nmodule load GCC/10.3.0\nmodule load OpenMPI/4.1.1\nmodule load TensorFlow/2.6.0-CUDA-11.3.1\nmodule load cuDNN/8.2.1.32-CUDA-11.3.1\n#\n#run the container with the runscript defined when we created it\n#nvidia-smi\n\npython LSTM_MO_AutoCV.py" + " " + str(split) + " " + emo + " " + phys + " " + rand + " " + valence
                        myBat.write(text)
                        myBat.close()
                        job_list.append('%s.sh' % filename)
    print("------- Jobs created: ", len(job_list), "-------")


    location = "/nfs/goldsaro/LSTM/"

    filename = "LSTMsubmit"
    myBat = open('%s.sh' % filename, 'wt')

elif (getcwd() == "/scale_wlg_persistent/filesets/home/goldsaro"): # NeSI
    for split in range(4, 8):
        for emo in ["cont", "6emo", "sum.PANAS", "indiv.PANAS"]:
            for phys in ["phys", "mocap", "both"]:
                for rand in ["shuf", "rand", "observed"]:
                    for valence in ["both", "pos", "neg"]:
                        name = "LSTM" + str(split) + "_" + str(emo)[:1].upper() + "_" + str(phys)[:1].upper() + "_" + str(rand)[:1].upper() + "_V" + str(valence)[:1].upper()
                        filename = location + name
                        myBat = open('%s.sl' % filename, 'wt')
                        text = "#!/bin/bash -e\n#\n#SBATCH --job-name=" + name + "\n" + "#SBATCH -o errout/" + name + ".out" + "\n" + "#SBATCH -e errout/" + name + ".err" + "\n" + "#SBATCH --time=1-00:00:00\n#SBATCH --gpus-per-node=" + gpu_type[phys] + ":1 \n#SBATCH --mem=128GB\n#\nmodule purge\nmodule load CUDA/11.6.2\nmodule load Python/3.10.5-gimkl-2022a\nmodule load TensorFlow/2.8.2-gimkl-2022a-Python-3.10.5\nmodule load cuDNN/8.4.1.50-CUDA-11.6.2\n#\n#nvidia-smi\n\npython LSTM_MO_AutoCV.py" + " " + str(split) + " " + emo + " " + phys + " " + rand + " " + valence
                        myBat.write(text)
                        myBat.close()
                        job_list.append('%s.sl' % filename)
    print("------- Jobs created: ", len(job_list), "-------")


    location = "/home/goldsaro/LSTM/"

    filename = "LSTMsubmit"
    myBat = open('%s.sl' % filename, 'wt')
                        
                        
# print(job_list)


text = "#!/bin/bash\n#\n#SBATCH --job-name=LSTMproc\n#SBATCH -o LSTMproc.out\n#SBATCH -e LSTMproc.err\n#SBATCH --time=00:30:00\n#SBATCH --mem=1GB\n#\n\n"

for file in job_list:
    text = text + "dos2unix " + file + "\n"

text = text + "\n"

for file in job_list:
    text = text + "sbatch " + file + "\n"

myBat.write(text)
myBat.close()


