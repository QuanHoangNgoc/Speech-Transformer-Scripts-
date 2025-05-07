import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Define input and output folders
# Replace with your input folder path
input_folder = "D:/BigDataset/raw"  # ! The data folder
# Replace with your output folder path
output_folder = os.path.join(os.getcwd(), "NP")

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate through all files in the input folder
sorted_list = sorted(os.listdir(input_folder))[-100:]  # ! Control
for idx, filename in enumerate(sorted_list):
    if filename.endswith(".wav"):
        # Load audio file
        file_path = os.path.join(input_folder, filename)
        y, sr = librosa.load(file_path)
        print(sr, end=" ")

        # # Compute mel spectrogram
        # mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        # mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # # Save mel spectrogram as a NumPy array
        # # Replace .wav extension with .npy
        # output_file_path = os.path.join(output_folder, filename[:-4] + ".npy")
        # np.save(output_file_path, mel_spectrogram_db)

        # if (idx % 100 == 0):
        #     print(f"Converted {filename} to {filename[:-4] + '.npy'}")
