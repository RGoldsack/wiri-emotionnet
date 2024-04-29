# Local Modules

# Installed Modules
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import datatable as dt
# Default Modules
import os
import pickle


def prepare_fold_data_loaders(full_loader, train_idx, val_idx, batch_size):
    """
    Prepare DataLoaders for training and validation sets for a specific fold.
    
    Args:
        full_loader (DataLoader): The DataLoader containing the full dataset.
        train_idx (list[int]): List of indices for the training set for the current fold.
        val_idx (list[int]): List of indices for the validation set for the current fold.
        batch_size (int): Batch size for DataLoader.
        
    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing the training and validation DataLoaders for the current fold.
    """
    # Extract the dataset from the full DataLoader
    full_dataset = full_loader.dataset
    
    # Create subsets for training and validation using the indices
    train_subset = torch.utils.data.Subset(full_dataset, train_idx)
    val_subset = torch.utils.data.Subset(full_dataset, val_idx)
    
    # Create DataLoaders for training and validation subsets
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=full_loader.collate_fn, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=full_loader.collate_fn, pin_memory=True)
    
    return train_loader, val_loader

class WiriDataset(Dataset):
    def __init__(self, directory, sequence_length=150):
        self.data = []
        self.labels = []
        self.emotions = []
        self.intensities = []
        
        for subdir, dirs, files in os.walk(directory):
            for filename in files:
                if filename.endswith(".pickle") and "Aligned" in subdir and "Allan" in filename:
                    file_path = os.path.join(subdir, filename)
                    with open(file_path, 'rb') as file:
                        content = pickle.load(file)
                        positions = np.array(content['positions'])
                        rotations = np.array(content['rotations'])
                        data = np.concatenate([positions, rotations[:, 1:]], axis=1)
                        hr_data = np.array(content['dramaturge_gsr'])  # Assuming this is HR data
                        
                        for i in range(0, len(data), sequence_length):
                            end_index = min(i + sequence_length, len(data))
                            if end_index - i < sequence_length:
                                continue
                            valid_indices = np.where(hr_data[i:end_index] <= 180)[0]
                            if len(valid_indices) < sequence_length:
                                continue
                            self.data.append(torch.tensor(data[i:end_index][valid_indices], dtype=torch.float32))
                            self.labels.append(torch.tensor(hr_data[i:end_index][valid_indices], dtype=torch.float32))
                            emotion, intensity = self.extract_info(filename)
                            self.emotions.append(emotion)
                            self.intensities.append(intensity)
    
    def extract_info(self, filename):
        parts = filename.split("_")
        emotion = parts[2]
        intensity = parts[3]
        return emotion, intensity
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.emotions[idx], self.intensities[idx]

class eMotionDataset(Dataset):
    def __init__(self, directory, sequence_length=150):
        self.data = []
        self.labels = []
        self.trials = []
        print(f"directory: {directory}")
        
        for subdir, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith("_dfBig.csv") and not file.startswith("._"):
                    print(f"file is {file}")
                    file_path = os.path.join(subdir, file)
                    data = dt.fread(file_path)
                    df = data.to_pandas()
                    df = df.dropna() 
                    valid_rows = df[df['HR'] <= 180]  # Filtering based on HR <= 180
                    motion_data = valid_rows.iloc[:, 1:358].values
                    hr_labels = valid_rows['HR'].values
                    trial_info = valid_rows['Trial'].values
                    
                    current_trial = None
                    for i, (row, hr, trial) in enumerate(zip(motion_data, hr_labels, trial_info)):
                        if trial != current_trial:
                            current_trial = trial
                            start_idx = i
                        if i - start_idx == sequence_length:
                            self.data.append(torch.tensor(motion_data[start_idx:i], dtype=torch.float32))
                            self.labels.append(torch.tensor(hr_labels[start_idx:i], dtype=torch.float32))
                            self.trials.append(trial)
                            start_idx = i

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.trials[idx], "no_intensity"

def collate_fn(batch):
    sequences, labels, emotions, intensities = zip(*batch)
    # Stack sequences and labels since they are already of fixed length
    return torch.stack(sequences), torch.stack(labels), emotions, intensities
