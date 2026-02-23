import os
import csv
from datetime import datetime, timedelta
from collections import defaultdict
import statistics
from scipy.stats import skew
import numpy as np
from multiprocessing import Pool

def column_data_factory():
    return np.array([], dtype=np.float32)

def interval_data_factory(active_window_length):
    return {
        'counts_10s': np.zeros(active_window_length // 10, dtype=np.int32),
        'column_data': defaultdict(column_data_factory)
    }

class IntervalDataFactory:
    def __init__(self, active_window_length):
        self.active_window_length = active_window_length

    def __call__(self):
        return {
            'counts_10s': np.zeros(self.active_window_length // 10, dtype=np.int32),
            'column_data': defaultdict(column_data_factory)
        }

def add_labels_from_csv(csv_file, csv_label_file, output_file):
    import pandas as pd
    label_df = pd.read_csv(csv_label_file)
    label_mapping = {}
    for _, row in label_df.iterrows():
        timestamp_str = row['Interval Start']
        timestamp = parse_timestamp(timestamp_str)
        label_mapping[timestamp] = {
            'first_frame_delay': row['First Frame Delay'],
            'buffer_empty_duration': row['Buffer Empty Duration'],
            'stall_duration': row['Stall Duration']
        }

    with open(csv_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        headers = next(reader)
        new_headers = ['label1_First_Frame_Delay', 'label2_Buffer_Empty_Duration', 'label3_Stall_Duration'] + headers
        writer.writerow(new_headers)

        for row in reader:
            timestamp_str = row[0]
            timestamp = parse_timestamp(timestamp_str)

            if timestamp in label_mapping:
                labels = label_mapping[timestamp]
                label_row = [
                    labels['first_frame_delay'],
                    labels['buffer_empty_duration'],
                    labels['stall_duration']
                ]
            else:
                label_row = ['', '', '']

            writer.writerow(label_row + row)

def parse_timestamp(timestamp_str):
    return datetime.strptime(timestamp_str, "%d/%b/%Y:%H:%M:%S")

def format_timestamp(timestamp):
    return timestamp.strftime("%d/%b/%Y:%H:%M:%S")

def calculate_entropy(values):
    if len(values) == 0:
        return 0.0
    counts, _ = np.histogram(values, bins='auto')
    counts = counts + 1e-10
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

def process_file(file_path, start_time, end_time, active_window_length, use_intermediate_files=True):
    if use_intermediate_files:
        import hashlib
        file_hash = hashlib.md5(file_path.encode()).hexdigest()
        intermediate_dir = os.path.join(os.path.dirname(file_path), 'temp_intermediate')
        os.makedirs(intermediate_dir, exist_ok=True)
        output_path = os.path.join(intermediate_dir, f'{file_hash}_win{active_window_length}.pkl')
        if os.path.exists(output_path):
            try:
                return load_intermediate_data(output_path)
            except:
                print(f"Failed to load intermediate file, reloading {file_path}")

    else:
        output_path = None

    local_data = defaultdict(IntervalDataFactory(active_window_length))

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            timestamp_str = row[0]
            timestamp = parse_timestamp(timestamp_str)
            if start_time <= timestamp <= end_time:
                interval_start = timestamp.replace(
                    minute=timestamp.minute - (timestamp.minute % 5),
                    second=0,
                    microsecond=0
                )
                if timestamp < interval_start + timedelta(seconds=active_window_length):
                    seconds_since = (timestamp - interval_start).total_seconds()
                    second_index = int(seconds_since // 10)
                    if second_index < active_window_length // 10:
                        local_data[interval_start]['counts_10s'][second_index] += 1
                        for i in range(31):
                            if i not in (0, 3):
                                try:
                                    value = float(row[i])
                                    local_data[interval_start]['column_data'][i] = np.append(
                                        local_data[interval_start]['column_data'][i], value
                                    )
                                except ValueError:
                                    pass
    if use_intermediate_files and output_path:
        save_intermediate_data(dict(local_data), output_path)
    return dict(local_data)

def save_intermediate_data(data_dict, file_path):
    import pickle
    with open(file_path, 'wb') as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_intermediate_data(file_path):
    import pickle
    try:
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)
        return data_dict
    except (EOFError, pickle.UnpicklingError, Exception) as e:
        print(f"Loading intermediate file {file_path} failed, will delete the corrupted file. Error message: {str(e)}")
        import os
        os.remove(file_path)
        raise

def calculate_g_values(data_dict, total_counts, g_history_length, active_window_length, sorted_intervals):
    g_data = {}
    for interval_start in sorted_intervals:
        prev_counts = []
        current_interval = interval_start
        for delta_seconds in [300 * i for i in range(1, g_history_length + 1)]:
            prev_time = current_interval - timedelta(seconds=delta_seconds)
            prev_counts.append(total_counts.get(prev_time, 0))
        g_data[interval_start] = prev_counts

    if sorted_intervals:
        sample_g_values = g_data[sorted_intervals[0]]
        col_means = []
        for col_idx in range(len(sample_g_values)):
            values = [g_data[interval][col_idx] for interval in sorted_intervals]
            non_zero = [v for v in values if v != 0]
            avg = sum(non_zero) / len(non_zero) if len(non_zero) > 0 else 0
            col_means.append(avg)

        for interval_start in sorted_intervals:
            for i in range(len(g_data[interval_start])):
                if g_data[interval_start][i] == 0:
                    g_data[interval_start][i] = col_means[i]

    return g_data

def process_csv_files(input_folders, output_file, enabled_groups, g_history_length, active_window_length=3,
                      use_intermediate_files=True,
                      csv_label_file=None, intermediate_dir=None):
    start_time = parse_timestamp("01/Jan/2024:00:00:00")
    end_time = parse_timestamp("31/Dec/2025:23:59:59")
    if intermediate_dir:
        os.makedirs(intermediate_dir, exist_ok=True)
        intermediate_file = os.path.join(intermediate_dir,
                                         os.path.basename(output_file) + f'.intermediate_win{active_window_length}.pkl')
    else:
        intermediate_file = output_file + f'.intermediate_win{active_window_length}.pkl'

    current_metadata_without_headers = {
        'input_folders': sorted(input_folders),
        'enabled_groups': sorted(enabled_groups),
        'g_history_length': g_history_length,
        'active_window_length': active_window_length,
        'use_intermediate_files': use_intermediate_files
    }

    if use_intermediate_files and os.path.exists(intermediate_file):
        print("Loading intermediate data...")
        try:
            loaded_data = load_intermediate_data(intermediate_file)
            if 'metadata' in loaded_data:
                existing_metadata = loaded_data['metadata']
                metadata_match = all(
                    existing_metadata.get(k) == v
                    for k, v in current_metadata_without_headers.items()
                )
                if metadata_match:
                    data_dict = loaded_data['data']
                    headers_list = existing_metadata.get('headers_list', [])
                    print("Intermediate file usable")
                else:
                    print("Intermediate file unusable")
                    os.remove(intermediate_file)
                    data_dict = None
            else:
                data_dict = None
        except:
            data_dict = None
    else:
        data_dict = None

    if not data_dict:
        files = []
        print(f"Collecting files from {input_folders}")
        for folder in input_folders:
            folder_files = []
            for root, dirs, filenames in os.walk(folder):
                sorted_files = sorted(
                    [os.path.join(root, f) for f in filenames if f.endswith('.csv')],
                    key=lambda x: os.path.basename(x)
                )
                folder_files.extend(sorted_files)
            files.extend(sorted(folder_files, key=lambda x: (
                next((i for i, f in enumerate(input_folders) if not os.path.relpath(x, f).startswith("..")), -1),
                x
            )))
            invalid_files = [f for f in folder_files if
                             not any(os.path.commonpath([f, in_folder]) == in_folder for in_folder in input_folders)]
            if invalid_files:
                print(f"Warning: {invalid_files}")

        if files:
            with open(files[0], 'r') as f:
                reader = csv.reader(f)
                headers_list = next(reader)
        else:
            headers_list = []

        with Pool(processes=31) as pool:
            results = pool.starmap(process_file,
                                   [(f, start_time, end_time, active_window_length, use_intermediate_files) for f in
                                    files])
        print("Processing complete.")
        print("Merging data...")
        data_dict = defaultdict(IntervalDataFactory(active_window_length))

        for res in results:
            for interval, data in res.items():
                current_counts = data_dict[interval]['counts_10s']
                new_counts = data['counts_10s']
                if len(current_counts) < len(new_counts):
                    current_counts = np.pad(current_counts, (0, len(new_counts) - len(current_counts)), 'constant')
                elif len(new_counts) < len(current_counts):
                    new_counts = np.pad(new_counts, (0, len(current_counts) - len(new_counts)), 'constant')
                data_dict[interval]['counts_10s'] = current_counts + new_counts
                for col, values in data['column_data'].items():
                    data_dict[interval]['column_data'][col] = np.concatenate(
                        (data_dict[interval]['column_data'][col], values)
                    )
        print("Merging complete.")

        for interval in data_dict:
            for col in data_dict[interval]['column_data']:
                data_dict[interval]['column_data'][col] = np.array(data_dict[interval]['column_data'][col])

        if use_intermediate_files:
            print("Saving intermediate data...")
            current_metadata = {
                **current_metadata_without_headers,
                'headers_list': headers_list
            }
            data_to_save = {
                'metadata': current_metadata,
                'data': data_dict
            }
            save_intermediate_data(data_to_save, intermediate_file)
            print("Intermediate data saved.")

    total_counts = {interval: sum(data['counts_10s']) for interval, data in data_dict.items()}
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        output_headers = ["Interval Start"]
        if 'A' in enabled_groups:
            output_headers.append("Record Count")
        if 'B' in enabled_groups:
            output_headers.extend([f"10s_{i}" for i in range(active_window_length // 10)])
        if 'C' in enabled_groups:
            for i in range(31):
                if i >=3:
                    col_name = headers_list[i]
                    output_headers.append(f"Mean_{col_name}")
        if 'D' in enabled_groups:
            for i in range(31):
                if i >=3:
                    col_name = headers_list[i]
                    output_headers.extend([f"Median_{col_name}", f"Variance_{col_name}", f"Skew_{col_name}"])
        if 'E' in enabled_groups:
            for i in range(31):
                if i >=3:
                    col_name = headers_list[i]
                    output_headers.append(f"Entropy_{col_name}")
        if 'F' in enabled_groups:
            for i in range(31):
                if i >=3:
                    col_name = headers_list[i]
                    for j in range(1, 6):
                        output_headers.append(f"Max{j}_{col_name}")
        if 'G' in enabled_groups:
            output_headers.extend([f"Prev{i}_Count" for i in range(1, g_history_length + 1)])
        if 'H' in enabled_groups:
            output_headers.append("TimeOfDay_Sec")
        writer.writerow(output_headers)

        sorted_intervals = sorted(data_dict.keys())
        for i in range(1, len(sorted_intervals)):
            if sorted_intervals[i] < sorted_intervals[i - 1]:
                sorted_intervals.sort()
                break

        g_data = {}
        if 'G' in enabled_groups:
            g_data = calculate_g_values(data_dict, total_counts, g_history_length, active_window_length,
                                        sorted_intervals)

        for interval_start in sorted_intervals:
            row = [format_timestamp(interval_start)]

            if 'A' in enabled_groups:
                row.append(sum(data_dict[interval_start]['counts_10s']))
            if 'B' in enabled_groups:
                counts_data = data_dict[interval_start]['counts_10s']
                if len(counts_data) < active_window_length // 10:
                    counts_data = np.pad(counts_data, (0, active_window_length // 10 - len(counts_data)), 'constant')
                row.extend(counts_data[:active_window_length // 10])
            if 'C' in enabled_groups:
                for i in range(31):
                    if i >=3:
                        values = data_dict[interval_start]['column_data'][i]
                        avg = np.mean(values) if len(values) else 0.0
                        row.append(avg)
            if 'D' in enabled_groups:
                for i in range(31):
                    if i >=3:
                        values = data_dict[interval_start]['column_data'][i]
                        if len(values):
                            median = np.median(values)
                            variance = np.var(values, ddof=1)
                            std_val = np.std(values, ddof=1)
                            if std_val == 0 or len(values) < 3:
                                skew_val = 0.0
                            else:
                                skew_val = skew(values, bias=False)
                        else:
                            median, variance, skew_val = 0, 0, 0
                        row.extend([median, variance, skew_val])
            if 'E' in enabled_groups:
                for i in range(31):
                    if i >=3:
                        values = data_dict[interval_start]['column_data'][i]
                        row.append(calculate_entropy(values) if len(values) > 0 else 0.0)
            if 'F' in enabled_groups:
                for i in range(31):
                    if i >=3:
                        values = data_dict[interval_start]['column_data'][i]
                        sorted_vals = np.sort(values)[-5:] if len(values) else np.array([])
                        sorted_vals = np.pad(sorted_vals, (0, 5 - len(sorted_vals)), 'constant', constant_values=0)
                        row.extend(sorted_vals.tolist())
            if 'G' in enabled_groups:
                row.extend(g_data[interval_start])

            if 'H' in enabled_groups:
                time_of_day_sec = interval_start.hour * 3600 + interval_start.minute * 60 + interval_start.second
                row.append(time_of_day_sec)
            writer.writerow(row)
    if csv_label_file and os.path.exists(csv_label_file):
        base_name = os.path.basename(output_file)
        labeled_output_file = os.path.join(
            os.path.dirname(output_file),
            'labeled_' + base_name
        )
        add_labels_from_csv(output_file, csv_label_file, labeled_output_file)
        print(f"Created: {labeled_output_file}")

if __name__ == "__main__":
    input_folders = [r".\QoS"]
    output_file = r".\file.csv"
    enabled_groups = {'A', 'C', 'D', 'G','H'}
    csv_label_file = r".\QoE.csv"
    intermediate_dir = r"./intermediate"
    process_csv_files(
        input_folders,
        output_file,
        enabled_groups,
        g_history_length=10,
        active_window_length=60,
        use_intermediate_files=False,
        csv_label_file=csv_label_file,
        intermediate_dir=intermediate_dir
    )
