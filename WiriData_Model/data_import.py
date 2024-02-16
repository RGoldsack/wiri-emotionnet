import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class EmotionDataset(Dataset):
    def __init__(self, directory, sequence_length=100):
        self.data = []
        self.labels = []
        self.sequence_lengths = []
        self.emotions = []
        self.intensities = []
        
        # Traverse subdirectories
        for subdir, dirs, files in os.walk(directory):
            for filename in files:
                if filename.endswith(".pickle") and "Aligned" in filename:
                    file_path = os.path.join(subdir, filename)
                    with open(file_path, 'rb') as file:
                        content = pickle.load(file)
                        positions = content['positions']
                        rotations = content['rotations']
                        data = np.concatenate([positions, rotations], axis=1)
                        
                        # Append data in chunks of sequence_length or less
                        for i in range(0, len(data), sequence_length):
                            end_index = min(i + sequence_length, len(data))
                            self.data.append(torch.tensor(data[i:end_index], dtype=torch.float))
                            self.labels.append(torch.tensor(content['actor_hr'][i:end_index], dtype=torch.float))
                            self.sequence_lengths.append(end_index - i)
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
        return self.data[idx], self.labels[idx], self.sequence_lengths[idx], self.emotions[idx], self.intensities[idx]

def collate_fn(batch):
    sequences, labels, lengths, emotions, intensities = zip(*batch)
    
    # Pad sequences and labels
    sequences_padded = pad_sequence(sequences, batch_first=True)
    labels_padded = pad_sequence(labels, batch_first=True)
    
    lengths_tensor = torch.tensor(lengths)
    
    # Note: emotions and intensities are lists of strings; no need to pad
    return sequences_padded, labels_padded, lengths_tensor, emotions, intensities

