"""
Data Preprocessing and Transformation Script for Jamming Detection

This script converts 1D RSSI signal data into 2D Mel spectrograms for use with a 2D CNN.
It loads the raw data, applies the transformation, and prepares train/test splits.
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import pickle


class RSSSIToMelSpectrogram:
    """Class to handle conversion of 1D RSSI signals to 2D Mel spectrograms."""
    
    def __init__(self, 
                 sampling_rate=1000, 
                 n_fft=256, 
                 hop_length=128, 
                 n_mels=64,
                 sequence_length=1000):
        """
        Initialize the preprocessing parameters.
        
        Args:
            sampling_rate (int): Sampling rate for the signal (Hz)
            n_fft (int): FFT window size
            hop_length (int): Hop length between windows
            n_mels (int): Number of Mel frequency bands
            sequence_length (int): Length of each signal sequence
        """
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sequence_length = sequence_length
        
    def signal_to_mel_spectrogram(self, signal):
        """
        Convert a 1D signal to a 2D Mel spectrogram.
        
        Args:
            signal (numpy.ndarray): 1D input signal
            
        Returns:
            numpy.ndarray: 2D Mel spectrogram in dB scale
        """
        # Ensure signal is the right length
        if len(signal) != self.sequence_length:
            if len(signal) > self.sequence_length:
                signal = signal[:self.sequence_length]
            else:
                # Pad with zeros if too short
                signal = np.pad(signal, (0, self.sequence_length - len(signal)), 'constant')
        
        # Convert to Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=signal.astype(np.float32),
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def batch_convert_signals(self, signals, verbose=True):
        """
        Convert a batch of 1D signals to 2D Mel spectrograms.
        
        Args:
            signals (numpy.ndarray): Array of 1D signals with shape (n_samples, sequence_length)
            verbose (bool): Whether to print progress
            
        Returns:
            numpy.ndarray: Array of 2D Mel spectrograms
        """
        n_samples = signals.shape[0]
        spectrograms = []
        
        for i, signal in enumerate(signals):
            if verbose and (i + 1) % 100 == 0:
                print(f"Processing signal {i + 1}/{n_samples}")
            
            mel_spec = self.signal_to_mel_spectrogram(signal)
            spectrograms.append(mel_spec)
        
        return np.array(spectrograms)


def load_rssi_data(data_dir):
    """
    Load RSSI data from text files.
    
    Args:
        data_dir (str): Directory containing the data files
        
    Returns:
        dict: Dictionary containing loaded data for each class
    """
    data_files = {
        'normal': os.path.join(data_dir, 'Rssi_Normal.txt'),
        'constant_jammer': os.path.join(data_dir, 'Rssi_CJ.txt'),
        'periodic_jammer': os.path.join(data_dir, 'Rssi_PJ.txt')
    }
    
    data = {}
    for class_name, file_path in data_files.items():
        if os.path.exists(file_path):
            print(f"Loading {class_name} data from {file_path}")
            data[class_name] = np.loadtxt(file_path)
            print(f"Loaded {len(data[class_name])} samples")
        else:
            print(f"Warning: File {file_path} not found!")
            
    return data


def prepare_sequences(data, sequence_length=1000):
    """
    Convert raw data into sequences of fixed length.
    
    Args:
        data (dict): Dictionary containing raw data for each class
        sequence_length (int): Length of each sequence
        
    Returns:
        tuple: (sequences, labels) where sequences is array of signal sequences
               and labels is array of corresponding class labels
    """
    sequences = []
    labels = []
    label_map = {'normal': 0, 'constant_jammer': 1, 'periodic_jammer': 2}
    
    for class_name, raw_data in data.items():
        # Calculate number of complete sequences
        n_sequences = len(raw_data) // sequence_length
        print(f"{class_name}: {n_sequences} sequences of length {sequence_length}")
        
        # Extract sequences
        class_sequences = raw_data[:n_sequences * sequence_length].reshape(
            n_sequences, sequence_length
        )
        
        sequences.append(class_sequences)
        labels.extend([label_map[class_name]] * n_sequences)
    
    # Combine all sequences
    all_sequences = np.concatenate(sequences, axis=0)
    all_labels = np.array(labels)
    
    return all_sequences, all_labels


def balance_dataset(sequences, labels, min_samples_per_class=None):
    """
    Balance the dataset by limiting each class to the same number of samples.
    
    Args:
        sequences (numpy.ndarray): Array of sequences
        labels (numpy.ndarray): Array of labels
        min_samples_per_class (int): If None, use the size of the smallest class
        
    Returns:
        tuple: (balanced_sequences, balanced_labels)
    """
    unique_labels = np.unique(labels)
    class_counts = [np.sum(labels == label) for label in unique_labels]
    
    if min_samples_per_class is None:
        min_samples_per_class = min(class_counts)
    
    print(f"Balancing dataset to {min_samples_per_class} samples per class")
    print(f"Original class distribution: {dict(zip(unique_labels, class_counts))}")
    
    balanced_sequences = []
    balanced_labels = []
    
    for label in unique_labels:
        class_indices = np.where(labels == label)[0]
        selected_indices = np.random.choice(
            class_indices, 
            size=min_samples_per_class, 
            replace=False
        )
        
        balanced_sequences.append(sequences[selected_indices])
        balanced_labels.extend([label] * min_samples_per_class)
    
    balanced_sequences = np.concatenate(balanced_sequences, axis=0)
    balanced_labels = np.array(balanced_labels)
    
    return balanced_sequences, balanced_labels


def visualize_mel_spectrogram(mel_spec, title="Mel Spectrogram", save_path=None):
    """
    Visualize a Mel spectrogram.
    
    Args:
        mel_spec (numpy.ndarray): 2D Mel spectrogram
        title (str): Title for the plot
        save_path (str): Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(
        mel_spec, 
        x_axis='time', 
        y_axis='mel', 
        sr=1000, 
        hop_length=128
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def main():
    """Main preprocessing pipeline."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Configuration
    config = {
        'sampling_rate': 1000,
        'n_fft': 256,
        'hop_length': 128,
        'n_mels': 64,
        'sequence_length': 1000,
        'test_size': 0.2,
        'data_dir': './Dataset/training'
    }
    
    print("=== RSSI to Mel Spectrogram Preprocessing ===")
    print(f"Configuration: {config}")
    
    # Load raw data
    print("\n1. Loading raw RSSI data...")
    raw_data = load_rssi_data(config['data_dir'])
    
    if not raw_data:
        print("Error: No data loaded. Please check data directory and files.")
        return
    
    # Prepare sequences
    print("\n2. Preparing sequences...")
    sequences, labels = prepare_sequences(raw_data, config['sequence_length'])
    print(f"Total sequences: {len(sequences)}, Labels: {len(labels)}")
    
    # Balance dataset
    print("\n3. Balancing dataset...")
    balanced_sequences, balanced_labels = balance_dataset(sequences, labels)
    
    # Initialize converter
    print("\n4. Converting to Mel spectrograms...")
    converter = RSSSIToMelSpectrogram(
        sampling_rate=config['sampling_rate'],
        n_fft=config['n_fft'],
        hop_length=config['hop_length'],
        n_mels=config['n_mels'],
        sequence_length=config['sequence_length']
    )
    
    # Convert to spectrograms
    spectrograms = converter.batch_convert_signals(balanced_sequences)
    print(f"Generated spectrograms shape: {spectrograms.shape}")
    
    # Visualize sample spectrograms
    print("\n5. Visualizing sample spectrograms...")
    class_names = ['Normal', 'Constant Jammer', 'Periodic Jammer']
    
    for class_idx in range(3):
        class_indices = np.where(balanced_labels == class_idx)[0]
        sample_idx = class_indices[0]
        
        visualize_mel_spectrogram(
            spectrograms[sample_idx],
            title=f"Sample {class_names[class_idx]} Mel Spectrogram",
            save_path=f"sample_{class_names[class_idx].lower().replace(' ', '_')}_spectrogram.png"
        )
    
    # Train-test split
    print("\n6. Creating train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        spectrograms, 
        balanced_labels, 
        test_size=config['test_size'], 
        random_state=42, 
        stratify=balanced_labels
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Spectrogram shape: {X_train.shape[1:]} (n_mels, time_steps)")
    
    # Add channel dimension for CNN (height, width, channels)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    print(f"Final training shape: {X_train.shape}")
    print(f"Final test shape: {X_test.shape}")
    
    # Save preprocessed data
    print("\n7. Saving preprocessed data...")
    os.makedirs('preprocessed_data', exist_ok=True)
    
    np.save('preprocessed_data/X_train.npy', X_train)
    np.save('preprocessed_data/X_test.npy', X_test)
    np.save('preprocessed_data/y_train.npy', y_train)
    np.save('preprocessed_data/y_test.npy', y_test)
    
    # Save configuration
    with open('preprocessed_data/config.pkl', 'wb') as f:
        pickle.dump(config, f)
    
    print("Preprocessing complete! Data saved to 'preprocessed_data/' directory")
    print(f"Class distribution in training set: {np.bincount(y_train)}")
    print(f"Class distribution in test set: {np.bincount(y_test)}")


if __name__ == "__main__":
    main()
