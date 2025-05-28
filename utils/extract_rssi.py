#!/usr/bin/env python3
"""
Script to extract RSSI values from FFT frames and save them
in the same format as normal_channel_paper.txt
"""

import numpy as np
import os
import struct

# File paths
input_file = '/home/omar/Downloads/SDR/PJ_RX.txt'
output_file = '/home/omar/Downloads/SDR/rssi_measurements_PJ_RX.txt'

file_size = os.path.getsize(input_file)

# Each FFT frame has 1024 float32 values (4 bytes each)
frame_size = 1024 * 4  # in bytes
num_frames = file_size // frame_size

print(f"File size: {file_size} bytes")
print(f"Each frame size: {frame_size} bytes")
print(f"Number of frames: {num_frames}")

# Open the output file
with open(output_file, 'w') as out_f:
    # Process each frame
    with open(input_file, 'rb') as in_f:
        for frame_idx in range(num_frames):
            # Read a single frame
            data = in_f.read(frame_size)
            if len(data) < frame_size:
                break  # Incomplete frame, stop processing
            
            # Convert binary data to numpy array of floats
            frame_data = np.array(struct.unpack('<1024f', data))
            
            # Calculate RSSI (Mean power across all frequency bins)
            # Since the data is already in dB, we take the mean directly
            rssi = np.mean(frame_data)
            
            # Alternative RSSI calculation methods:
            # 1. Maximum power in any frequency bin
            # rssi_max = np.max(frame_data)
            
            # 2. Sum of power across all frequency bins (convert from dB first)
            # linear_power = 10**(frame_data/10)
            # total_power = np.sum(linear_power)
            # rssi_sum = 10 * np.log10(total_power)
            
            # Write RSSI value to the output file
            out_f.write(f"{rssi:.4f}\n")
            
            # Print progress every 500 frames
            if frame_idx % 500 == 0:
                print(f"Processed {frame_idx}/{num_frames} frames...")

print(f"Conversion complete. RSSI measurements saved to {output_file}")
print(f"Each line in the output file contains a single RSSI value.")
