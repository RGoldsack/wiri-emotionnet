from data_import import *
from model import *

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data.dataset import random_split
from data_import import EmotionDataset
from model import CustomModel
from scipy.stats import pearsonr
import numpy as np
import csv


# Parameters
n_inputs = 306  # Adjust based on your data
hidden_nodes = 128  # Example value
dropout = 0.2
runtimeOptions = [None, 32, None, False]  # Example values; adjust as necessary
learning_rate = 1e-3
epochs = 1000  # As per your request; consider starting smaller
batch_size = 512
sequence_length = 100  # Adjust based on your sequence length preference

# Initialize dataset
dataset_directory = "/Volumes/fastt/Wiri/Performance_capture/"
full_dataset = EmotionDataset(dataset_directory, sequence_length=100)

# Split dataset
train_size = int(0.7 * len(full_dataset))
test_size = int(0.15 * len(full_dataset))
validation_size = len(full_dataset) - train_size - test_size
train_dataset, test_dataset, validation_dataset = random_split(full_dataset, [train_size, test_size, validation_size])

# Create dataloaders for each set
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
# Initialize model
model = CustomModel(n_inputs, hidden_nodes, dropout, runtimeOptions)
model.train()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# Training loop
def validate(model, validation_loader, criterion):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():  # No need to track gradients during validation
        for sequences, labels, lengths, _, _ in validation_loader:
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * sequences.size(0)
            total_samples += sequences.size(0)
    
    avg_loss = total_loss / total_samples
    return avg_loss

# Training loop with validation
for epoch in range(epochs):
    model.train()  # Set model to training mode
    total_train_loss = 0.0
    total_train_samples = 0
    
    for sequences, labels, lengths, _, _ in train_loader:
        optimizer.zero_grad()
        outputs = model(sequences, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * sequences.size(0)
        total_train_samples += sequences.size(0)
    
    avg_train_loss = total_train_loss / total_train_samples
    avg_val_loss = validate(model, validation_loader, criterion)
    
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

print("Training completed.")

# Evaluation function
def evaluate_model_and_save_results(loader, file_path):
    model.eval()
    predictions, actuals, emotions, intensities = [], [], [], []
    with torch.no_grad():
        for inputs, labels, lengths, emotion, intensity in loader:
            optimizer.zero_grad()
            outputs = model(inputs.float(), lengths)
            outputs = model(inputs.float())
            predictions.extend(outputs.numpy())
            actuals.extend(labels.numpy())
            emotions.extend(emotion)
            intensities.extend(intensity)
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculating metrics per sample
    mse_per_sample = np.mean((actuals - predictions) ** 2, axis=1)
    correlation_per_sample = [pearsonr(a, p)[0] for a, p in zip(actuals, predictions)]
    
    # Save to CSV
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Emotion', 'Intensity', 'MSE', 'Correlation'])
        for i in range(len(predictions)):
            writer.writerow([emotions[i], intensities[i], mse_per_sample[i], correlation_per_sample[i]])
    
    print(f"Results saved to {file_path}.")

# Example usage
evaluate_model_and_save_results(train_loader,      "./Results/train_results.csv")
evaluate_model_and_save_results(validation_loader, "./Results/val_results.csv")
evaluate_model_and_save_results(test_loader,       "./Results/test_results.csv")


