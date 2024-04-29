# Local Modules
from data_import import WiriDataset, eMotionDataset, collate_fn, prepare_fold_data_loaders
from model import *
from utils import *
# Installed Modules
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pandas as pd
# Default Modules
import csv
import sys
import time
import glob
import argparse
import warnings

warnings.filterwarnings("ignore")


def get_params():
    global params
    params = {
        "batch_size": 256,
        "epochs_final": 1000,
        "n_splits": 5,
        "sequence_length": 150,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "learning_rate": 1e-3,
        "hidden_nodes": 128,
        "dropout": 0.2,
        "early_stopping_patience": 10,
        "early_stopping_delta" : 0.1,
        "checkpoint_interval": 50,
        "num_cpus" : 16
    }

def find_latest_checkpoint(checkpoint_dir, fold, dataset):
    latest_epoch = 0
    latest_checkpoint = None
    checkpoint_pattern = f"{checkpoint_dir}/{dataset}_checkpoint_fold_{fold}_epoch_*.pt"
    checkpoints = glob.glob(checkpoint_pattern)
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        latest_epoch = int(os.path.basename(latest_checkpoint).split('_epoch_')[1].split('.pt')[0])
    return latest_epoch, latest_checkpoint

def train_with_hyperparameters(train_loader, val_loader, epochs, fold, checkpoint_dir):
    ncols = 306 if params["dataset"] == "Wiri" else 357
    model = CustomModel(n_inputs=ncols, hidden_nodes=params["hidden_nodes"], dropout=params["dropout"]).to(params["device"])
    optimizer = Adam(model.parameters(), lr=params["learning_rate"])
    criterion = nn.MSELoss()
    early_stopping_counter = 0

    
    start_epoch, latest_checkpoint = find_latest_checkpoint(checkpoint_dir, fold, params["dataset"])
    if latest_checkpoint is not None:
        checkpoint = torch.load(latest_checkpoint, map_location=params["device"])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Resuming fold {fold} and epoch {start_epoch + 1}")

    time_per_epoch_list = []
    for epoch in range(start_epoch, epochs):
        t_epoch = time.time()
        total_loss = 0.0
        best_val_loss = 999999
        model.train()
        model.to(params["device"])
        for motion, hr, _, _ in train_loader:
            motion, hr = motion.to(params["device"]), hr.to(params["device"])
            optimizer.zero_grad()
            outputs = model(motion)

            loss = criterion(outputs.squeeze(-1).to(params["device"]), hr)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * motion.size(0)
        
        avg_train_loss = total_loss / len(train_loader.dataset)
        avg_val_loss = validate(model, val_loader, criterion)

        # Logging the training progress
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f} " +  
              time_calc(t_epoch, string = "| Current Epoch Took") + 
              ", " + estimate_time_remaining(t_epoch, epoch, epochs, time_per_epoch_list) + 
              " | " + gpu_usage(), flush=True)

        # Save checkpoint
        checkpoint_save_path = f"{checkpoint_dir}/{params['dataset']}_checkpoint_fold_{fold}_epoch_{epoch}.pt"
        if (epoch + 1) % params["checkpoint_interval"] == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1
            }, checkpoint_save_path)
            print(f"Checkpoint saved at fold {fold}, epoch {epoch + 1}")

        # Early stopping condition
        if (avg_train_loss < avg_val_loss) and epoch > 100:
            early_stopping_counter += 1
            print("counter goes up!")
            if early_stopping_counter >= params["early_stopping_patience"]:
                print(f"Early stopping triggered at epoch {epoch + 1}. Training loss has been lower than validation loss for {params['early_stopping_patience']} consecutive epochs.")
                break
        else:
            early_stopping_counter = 0  # Reset counter if condition does not hold

    return model

def cross_validate_model(train_val_loader, test_loader, dataset = ""):
    checkpoint_dir = os.path.join(get_data_directory(), "Wiri/Performance_capture/MiniModel_Results/", "checkpoints/")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    kf = KFold(n_splits=params["n_splits"], shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(train_val_loader.dataset)))):
        print(f"Starting fold {fold + 1}/{params['n_splits']}")
        # Prepare data loaders for the fold
        train_loader, val_loader = prepare_fold_data_loaders(train_val_loader, train_idx, val_idx, batch_size=params["batch_size"])
        
        # Train and evaluate the model
        model = train_with_hyperparameters(train_loader, val_loader, params["epochs_final"], fold, checkpoint_dir)
        evaluate_model_and_save_results(model, val_loader, f"fold_{fold}_val_{dataset}")
        evaluate_model_and_save_results(model, test_loader, f"fold_{fold}_test_{dataset}")
        evaluate_model_and_save_results(model, train_loader, f"fold_{fold}_train_{dataset}")

def split_dataset(full_loader):
    dataset_size = len(full_loader.dataset)
    test_split = int(np.floor(0.1 * dataset_size))
    train_val_size = dataset_size - test_split
    train_val_dataset, test_dataset = random_split(full_loader.dataset, [train_val_size, test_split], generator=torch.Generator().manual_seed(42))
    return train_val_dataset, test_dataset

def validate(model, validation_loader, criterion):
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for motion, hr, _, _ in validation_loader:
            motion, hr = motion.to(params["device"]), hr.to(params["device"])
            model = model.to(params["device"])
            outputs = model(motion)
            loss = criterion(outputs.squeeze(), hr)
            total_val_loss += loss.item() * motion.size(0)
    
    avg_val_loss = total_val_loss / len(validation_loader.dataset)
    return avg_val_loss

def evaluate_model_and_save_results(model, loader, prefix):
    results_folder = get_data_directory() + "Wiri/Performance_capture/MiniModel_Results/"
    batch_statistics_path = os.path.join(results_folder, f"{prefix}_batch_statistics.csv")
    detailed_results_path = os.path.join(results_folder, f"{prefix}_detailed_results.csv")

    model.eval()
    all_actuals, all_predictions, emo_l, int_l, cor_l, mse_l = [], [], [], [], [], []

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    with torch.no_grad():
        for motion, hr, emotion, intensity in loader:
            outputs      = model(motion.to(params["device"])).squeeze().cpu().numpy()
            actuals      = hr.cpu().numpy()
            all_predictions.extend(outputs)
            all_actuals.extend(actuals)

            correlations, mse = safe_stats(actuals, outputs)
            emo_l.extend(list(emotion))
            int_l.extend(list(intensity))
            cor_l.extend(correlations)
            mse_l.extend(mse)

    batch_stats = pd.DataFrame({
        "Emotion": np.array(emo_l),
        "Intensity": np.array(int_l),
        "Correlation": np.array(cor_l),
        "MSE": np.array(mse_l)
    })
    batch_stats.to_csv(batch_statistics_path, index=False)

    all_predictions = np.array(all_predictions).flatten()
    all_actuals = np.array(all_actuals).flatten()
    all_results = pd.DataFrame({
        "Actual HR": all_actuals,
        "Predicted HR": all_predictions
    })
    all_results.to_csv(detailed_results_path, index=False)
    
    print(f"Detailed results saved to {detailed_results_path}.")
    print(f"Batch statistics saved to {batch_statistics_path}.")

    # Now let's plot the predicted vs actual HR
    plt.figure(figsize=(10, 5))
    plt.scatter(all_actuals, all_predictions, alpha=0.5)  # Plot with actual HR on x-axis and predicted HR on y-axis
    plt.title('Predicted HR vs Actual HR')
    plt.xlabel('Actual Heart Rate (HR)')
    plt.ylabel('Predicted Heart Rate (HR)')
    plt.grid(True)

    # Save the plot before showing it
    plot_path = os.path.join(results_folder, f"{prefix}_actual_vs_predicted.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}.")

    # Now display the plot
    plt.show()
    plt.close()  # Close the plot to avoid displaying it inline if not necessary


def safe_stats(actuals, predictions):
    def cor(actuals, predictions):
        try:
            corr, _ = pearsonr(actuals, predictions)
            return corr if not np.isnan(corr) else 0.0  # Return 0 correlation if result is NaN
        except:
            return 0.0  # In case of an error (e.g., constant series), return 0 correlation
    correl, mse = [], []
    for i in range(actuals.shape[0]):
        try:
            correl.append(cor(  actuals[i, :],  predictions[i, :]))
            mse.append(np.mean((actuals[i, :] - predictions[i, :]) ** 2))
        except:
            correl.append(cor(  actuals[i],  predictions[i]))
            mse.append(np.mean((actuals[i] - predictions[i]) ** 2))
    return correl, mse
    
def get_dataloader(dataset_name, directory):
    if dataset_name == 'Wiri':
        dataset = WiriDataset(directory + "Wiri/Performance_capture/", params["sequence_length"])
    elif dataset_name == 'eMotion':
        try:
            dataset = eMotionDataset(directory + "DyadFiles/", params["sequence_length"])
        except:
            dataset = eMotionDataset(directory + "Data/DyadFiles/", params["sequence_length"])
    else:
        raise ValueError("Unknown dataset name")
    return DataLoader(dataset, batch_size=params["batch_size"], shuffle=True, collate_fn=collate_fn)

def main(argv = sys.argv[1:]):
    # Process Arguments
    parser = argparse.ArgumentParser(
        prog=__file__,
        description="""Pre-Process Wiri Data""",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-d", "--dataset", type = str,  default = "Wiri", help = "Dataset being run.")

    args = vars(parser.parse_args(argv))
    get_params()
    params["dataset"] = args["dataset"]
    # Load data
    print(f"-- Loading data for {params['dataset']} dataset.. --")
    dataset_directory = get_data_directory()
    data_loaders = get_dataloader(params["dataset"], dataset_directory)
    train_val_dataset, test_dataset = split_dataset(data_loaders)
    train_val_loader = DataLoader(train_val_dataset, batch_size=params["batch_size"], shuffle=True,  collate_fn=collate_fn, num_workers=params["num_cpus"], pin_memory=True)
    test_loader      = DataLoader(test_dataset,      batch_size=params["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=params["num_cpus"], pin_memory=True)

    print("-- Data loaded --")

    # Perform n-fold cross-validation with hyperparameter tuning and evaluation
    print(f"-- Beginning Cross Validation with {params['n_splits']} splits --")
    cross_validate_model(train_val_loader, test_loader, params["dataset"])

if __name__ == "__main__":
    exit_code = int(not main())
    sys.exit(exit_code)
