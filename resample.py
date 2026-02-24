import json
import math

INPUT_FILE = "gesture_data.json"
OUTPUT_FILE = "gesture_data_resampled.json"
TARGET_LENGTH = 100  # number of samples per gesture

def resample_samples(samples, target_length):
    """
    Resample a list of samples to target_length using simple indexing.
    """
    if len(samples) == target_length:
        return samples.copy()
    elif len(samples) == 0:
        return []
    
    resampled = []
    for i in range(target_length):
        # map i to index in original array
        idx = int(i * len(samples) / target_length)
        if idx >= len(samples):
            idx = len(samples) - 1
        resampled.append(samples[idx])
    return resampled

def resample_gesture_data(data, target_length):
    """
    Resample all recordings in the dataset.
    """
    new_data = []
    for entry in data:
        gesture = entry["gesture"]
        samples = entry["samples"]
        resampled_samples = resample_samples(samples, target_length)
        new_data.append({
            "gesture": gesture,
            "samples": resampled_samples
        })
    return new_data

def main():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)
    
    resampled_data = resample_gesture_data(data, TARGET_LENGTH)
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(resampled_data, f, indent=2)
    
    print(f"Resampled dataset saved to {OUTPUT_FILE} with {TARGET_LENGTH} samples per recording.")

if __name__ == "__main__":
    main()