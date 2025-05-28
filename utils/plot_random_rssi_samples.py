#!/usr/bin/env python3
"""
Script to plot a random consecutive 1000 samples from the RSSI measurements file
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import os

# File path
input_file = '/home/omar/Downloads/SDR/rssi_measurements_CJ_RX.txt'

# Check if input file exists
if not os.path.exists(input_file):
    print(f"Error: Input file {input_file} does not exist.")
    exit(1)

# Read all RSSI values from the file
with open(input_file, 'r') as f:
    lines = f.readlines()

# Convert text values to float
rssi_values = [float(line.strip()) for line in lines]

# Get total number of samples
total_samples = len(rssi_values)
print(f"Total RSSI samples in the file: {total_samples}")

# Check if we have at least 1000 samples
if total_samples < 1000:
    print(f"Warning: File contains only {total_samples} samples, which is less than 1000.")
    num_samples = total_samples
else:
    num_samples = 1000

# Choose a random starting point (ensure we can get 1000 consecutive samples)
max_start_index = total_samples - num_samples
if max_start_index < 0:
    start_index = 0
else:
    start_index = random.randint(0, max_start_index)

# Get 1000 consecutive samples from the random starting point
selected_samples = rssi_values[start_index:start_index + num_samples]

# Create x-axis (sample numbers)
x_values = np.arange(start_index, start_index + num_samples)

# Plot the samples
plt.figure(figsize=(12, 6))
plt.plot(x_values, selected_samples, 'b-')
plt.grid(True)
plt.xlabel('Sample Index')
plt.ylabel('RSSI (dB)')
plt.title(f'Random {num_samples} Consecutive RSSI Samples (Starting at index {start_index})')

# Add horizontal line at mean value
mean_rssi = np.mean(selected_samples)
plt.axhline(y=mean_rssi, color='r', linestyle='--', 
           label=f'Mean RSSI: {mean_rssi:.2f} dB')

# Add horizontal lines at standard deviation boundaries
std_rssi = np.std(selected_samples)
plt.axhline(y=mean_rssi + std_rssi, color='g', linestyle=':', 
           label=f'Mean + Std Dev: {mean_rssi + std_rssi:.2f} dB')
plt.axhline(y=mean_rssi - std_rssi, color='g', linestyle=':', 
           label=f'Mean - Std Dev: {mean_rssi - std_rssi:.2f} dB')

# Add min and max values as annotations
min_rssi = np.min(selected_samples)
max_rssi = np.max(selected_samples)
plt.annotate(f'Min: {min_rssi:.2f} dB', 
            xy=(x_values[np.argmin(selected_samples)], min_rssi),
            xytext=(10, -30), textcoords='offset points',
            arrowprops=dict(arrowstyle='->'))
plt.annotate(f'Max: {max_rssi:.2f} dB', 
            xy=(x_values[np.argmax(selected_samples)], max_rssi),
            xytext=(10, 30), textcoords='offset points',
            arrowprops=dict(arrowstyle='->'))

plt.legend()
plt.tight_layout()

# Save the plot
output_file = '/home/omar/Downloads/SDR/rssi_samples_plot.png'
plt.savefig(output_file)
print(f"Plot saved as {output_file}")

# Optionally show the plot if running in an interactive environment
plt.show()

# Also print some statistics
print("\nStatistics for the selected samples:")
print(f"Mean RSSI: {mean_rssi:.4f} dB")
print(f"Standard Deviation: {std_rssi:.4f} dB")
print(f"Minimum RSSI: {min_rssi:.4f} dB")
print(f"Maximum RSSI: {max_rssi:.4f} dB")
print(f"Range (Max-Min): {max_rssi - min_rssi:.4f} dB")
