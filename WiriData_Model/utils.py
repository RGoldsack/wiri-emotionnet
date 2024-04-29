# Local Modules

# Installed Modules
import torch
# Default Modules
import time
import subprocess
import os
import sys

def get_data_directory():
    if ("/nfs/home/goldsaro" in os.getcwd()):
        return "/nfs/scratch/goldsaro/"
    elif r"C:\Users\golds\OneDrive\Documents\GitHub" in os.getcwd():
        return "C:/Users/golds/OneDrive/Desktop/Performance_capture/"
    elif "/Users/roydon/Documents/git" in os.getcwd():
        return "/Volumes/fastt/"
    
def time_calc(t0, string = "Time Elapsed"):
    t1 = time.time()
    hours, rem = divmod(t1 - t0, 3600)
    minutes, seconds = divmod(rem, 60)
    return string + ": {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)

def gpu_usage():
    try:
        num_gpus = torch.cuda.device_count()
        gpu_info = []

        for gpu_id in range(num_gpus):
            result = subprocess.check_output([
                "nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits", f"--id={gpu_id}"
            ])
            gpu_usage, vram_usage = map(int, result.decode("utf-8").strip().split(','))
            gpu_info.append(f"GPU {gpu_id} Usage: {gpu_usage}% | VRAM: {vram_usage} MB")

        return ' | '.join(gpu_info)
    except Exception as e:
        return "Unable to retrieve GPU usage."

def estimate_time_remaining(t0, epoch, total_epochs, time_per_epoch_list):
    time_per_epoch_list.append(time.time() - t0)
    if len(time_per_epoch_list) > 50:
        time_per_epoch_list.pop(0)

    avg_time_per_epoch = sum(time_per_epoch_list) / len(time_per_epoch_list)

    epochs_left = total_epochs - epoch - 1
    time_left = epochs_left * avg_time_per_epoch
    hours_left, remainder = divmod(time_left, 3600)
    minutes_left, seconds_left = divmod(remainder, 60)

    return f"Time left: {int(hours_left)}h {int(minutes_left)}m {int(seconds_left)}s"


def quiet(func, *args, **kwargs):
    save_stdout = sys.stdout
    sys.stdout = open('trash', 'w')
    out = func(*args, **kwargs)
    sys.stdout = save_stdout
    if out is not None:
        return None
