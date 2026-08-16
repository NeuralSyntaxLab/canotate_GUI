"""
This module will get model as input along with directory of wav files and will generate a Keinan-based format annotation mat file
"""

from datetime import date
import os
import torch
import numpy as np
import json
from bird_song_detector.model_zoo import ConvMLP
from bird_song_detector.nn_segmentation.segmentation_utils import _merge_close_segments, _get_assential_info_from_filename
# Upstream pulls these from data_loader.py and training_loop.py, which drag in
# h5py/pandas/sklearn/seaborn/tensorboard. Both functions are reproduced
# verbatim in _helpers.py instead -- see PROVENANCE.md.
from bird_song_detector._helpers import load_from_wav, normalize_per_segment

class InferenceToAnnotationData:

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.recording_fs = None

    def translate_inference_to_annotation_format(self, inference_output, fs, segment_size=256):
        """
        annotation format is time_start, time_end, label (999 as placeholder for now because we only segment)
        inputs- inference_output: 1d array of binary values (0 or 1) indicating presence of bird song
                fs: sampling frequency of the original audio file
        output- annotation_data: list of lists of size 3 where each inner list is [time_start, time_end, label]
        """
        annotation_data = []
        in_segment = False
        segment_start = 0
        for i, value in enumerate(inference_output):
            if value == 1 and not in_segment:
                in_segment = True
                segment_start = i
            elif value == 0 and in_segment:
                in_segment = False
                segment_end = i
                time_start = (segment_start / fs) * segment_size  # Convert segment index to time in seconds
                time_end = (segment_end / fs) * segment_size
                annotation_data.append([time_start, time_end, 999])  # 999 as placeholder label

        # Handle case where the last segment goes until the end of the audio
        if in_segment:
            time_start = (segment_start / fs) * segment_size
            time_end = (len(inference_output) / fs) * segment_size
            annotation_data.append([time_start, time_end, 999])

        return _merge_close_segments(annotation_data)

    def _is_valid_date(self, date_parts):
        if len(date_parts) != 3:
            return False
        try:
            year, month, day = map(int, date_parts)
            date(year, month, day)  
            return True
        except ValueError:
            return False

    def _extract_date_from_filename(self, filename):
        # Assuming filename format is something like "birdname_idx_year_month_day_hour_min_sec.wav"
        base_name = os.path.splitext(filename)[0]  # Remove extension
        parts = base_name.split('_')  # Split by underscore
        date_parts = parts[-6:-3]  # Get the last 6 - 3 parts which should be year, month, day
        if self._is_valid_date(date_parts):
                return '-'.join(date_parts)
        return "unknown_date"

    def generate_annotation_data(self, wav_dir, output_dir):
        # This function will process each wav file in the directory, run inference using the model,
        # and save the results in the Keinan-based format annotation mat file in the output directory.
        # walk through wav_dir, for each wav file, run inference and save annotation data
        all_wav_files = [os.path.join(r, f) for r, _, fs in os.walk(wav_dir) for f in fs if f.endswith('.wav')]
        print(f"Found {len(all_wav_files)} wav files to process in '{wav_dir}'")
        results = {}
        for root, dirs, files in os.walk(wav_dir):
            for file in files:
                if file.endswith('.wav'):
                    current_day = self._extract_date_from_filename(file)
                    birds_name = file.split('_')[0]  # Assuming bird name is the first part of the filename
                    if current_day not in results:
                        results[current_day] = []
                    wav_path = os.path.join(root, file)
                    try:
                        annotation_data = self.annotatate_data(wav_path)
                        results[current_day].append({"file_name": file, "segments": annotation_data})
                    except Exception as e:
                        print(f"Error processing {file}: {e}")
        # Save results to output directory
        if results:
            self.save_annotation_data(results, output_dir, birds_name)
                    
    def annotatate_data(self, wav_path):
        predictions = self.run_inference(wav_path)
        annotation_data = self.translate_inference_to_annotation_format(predictions, self.recording_fs)
        return annotation_data


    def run_inference(self, wav_path, confidence_threshold=0.5, batch_size=2048):
        data, _, indices, metadata = load_from_wav(wav_path)
        if self.recording_fs is None:
            self.recording_fs = metadata['fs']
        self.model.eval()
        all_predictions = []
        with torch.no_grad():
            outputs = self.model(normalize_per_segment(data.float()).to(self.device))
            preds = (torch.sigmoid(outputs).squeeze(-1) > confidence_threshold).cpu().numpy().astype(int)
            all_predictions.append(preds)
        return np.concatenate(all_predictions)
    
    def save_annotation_data(self, annotation_data, output_dir, filename):
        output_file_path = os.path.join(output_dir, filename + '_segmented.json')
        with open(output_file_path, 'w') as f:
            json.dump(annotation_data, f, indent=4)
        print(f"Saved annotation data to {output_file_path}")




def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run bird song inference and generate annotation data.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model weights file.")
    parser.add_argument("--wav_path", type=str, default=None, help="Directory containing the wav files.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the annotation data.")
    args = parser.parse_args()

    model_architecture = ConvMLP(input_size=256)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') if not torch.cuda.is_available() else torch.device('cuda')

    model_weights_path = args.model_path or input("Enter the path to the model weights file: ")
    if not os.path.exists(model_weights_path):
        print(f"Model weights file not found at '{model_weights_path}'. Please check the path and try again.")
        exit(1)
    model_weights = torch.load(model_weights_path, map_location=device)
    model_architecture.load_state_dict(model_weights)
    print("Model weights loaded successfully.")

    wav_dir = args.wav_path or input("Enter the directory containing the wav files: ")
    output_dir = args.output_dir or input("Enter the directory to save the annotation data: ")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    inference_to_annotation = InferenceToAnnotationData(model_architecture, device)
    inference_to_annotation.generate_annotation_data(wav_dir, output_dir)


if __name__ == "__main__":
    main()