#!/usr/bin/env python3
"""
Script to extract raw FFT bin values from FFT frames, compute the mean of every 40 values,
and save each mean on a line in the output text file.
"""

import numpy as np
import os
import struct
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract FFT bin values from binary file')
    parser.add_argument('--input', '-i', type=str, required=False,
                       default='/home/omar/Downloads/SDR/Dataset/USRP_Files_Binary/PJ_RX.txt',
                       help='Input binary file path')
    parser.add_argument('--output', '-o', type=str, required=False,
                       default=None,
                       help='Output text file path (default: auto-generated based on input)')
    parser.add_argument('--num-frames', '-n', type=int, required=False,
                       default=None,
                       help='Number of frames to process (default: process all frames)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # File paths
    input_file = args.input
    
    # Auto-generate output file name if not provided
    if args.output is None:
        base_name = os.path.basename(input_file)
        name_without_ext = os.path.splitext(base_name)[0]
        output_file = os.path.join(os.path.dirname(input_file), f"fft_means_4bins_{name_without_ext}.txt")
    else:
        output_file = args.output

    file_size = os.path.getsize(input_file)

    # Each FFT frame has 1024 float32 values (4 bytes each)
    frame_size = 1024 * 4  # in bytes
    num_frames = file_size // frame_size
    
    # Limit number of frames if specified
    if args.num_frames is not None and args.num_frames < num_frames:
        num_frames = args.num_frames
        print(f"Processing only {num_frames} frames as requested")

    print(f"File size: {file_size} bytes")
    print(f"Each frame size: {frame_size} bytes")
    print(f"Number of frames to process: {num_frames}")
    print(f"Each frame contains 1024 FFT bin values")

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
                
                # Calculate mean for every 40 consecutive values
                for i in range(0, len(frame_data), 40):
                    # Get the next 40 values and calculate their mean
                    chunk = frame_data[i:i+40]
                    mean_value = np.mean(chunk)
                    out_f.write(f"{mean_value:.6f}\n")
                
                # Print progress every 100 frames
                if frame_idx % 100 == 0:
                    print(f"Processed {frame_idx}/{num_frames} frames...")

    print(f"Conversion complete. FFT bin values saved to {output_file}")
    print(f"Each line in the output file contains the mean of 40 consecutive FFT bin values.")
    print(f"Total lines in output file: {num_frames * (1024 // 40)}")

if __name__ == "__main__":
    main()
