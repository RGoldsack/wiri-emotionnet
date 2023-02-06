# wiri-emotionnet
Repository containing code used for my Masters Thesis project.

## Structure

The MachineLearning branch includes the 'LSTM_MO_AutoCV.py' file which contains the various functions to handle the execution and running of the various machine learning models. This file requires the 'Import.py' file to also be in the same directory as at runtime it executes 'model_import()' which imports the relevant data for the current model.

The 432 machine learning models (if excluding models trained/tested on shuffled observed data) are run embarrassingly parrallel using `LSTMsubmit.py` which is executed by `LSTMprep.sh`. `LSTMsubmit.py` generates a the requires bash files for running each of the different models and creates the 'LSTMsubmit.sh' file which executes all of the generated job files that are contained within the LSTM folder (at directory where `LSTMsubmit.py' was run. These jobs are then submitted to the Raapoi High Performance Compute (HPC) GPU nodes. 
