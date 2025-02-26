from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd

all_index_sequence = None
count = 0

def split_data(df, num_sets=5, first_set_ratio=0.3):
    """
    Split the dataset into specified number of sets. The first set will have the
    proportion specified by 'first_set_ratio', and the remaining data is evenly
    split among the rest of the sets.
    """
    num_samples = len(df)
    global all_index_sequence
    all_index_sequence = np.arange(0, num_samples)
    first_set_size = int(num_samples * first_set_ratio)
    remaining_set_size = (num_samples - first_set_size) // (num_sets - 1)

    datasets = []
    start_idx = 0
    for i in range(num_sets):
        if i == 0:
            end_idx = start_idx + first_set_size
        else:
            end_idx = start_idx + remaining_set_size

        subset = df.iloc[start_idx:end_idx]
        datasets.append(subset)
        start_idx = end_idx

    return datasets


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=True, scaler=None):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    global all_index_sequence, count
    index_sequence = all_index_sequence[count:count+num_samples]
    count += num_samples

    # 将DataFrame的索引转换为数字序列

    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        # time_in_index = np.tile(index_sequence, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
        # data_list.append(time_in_index)

    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 1))
        # day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        day_in_week[:, :, 0] = np.array(df.index.dayofweek).reshape(-1, 1)
        data_list.append(day_in_week)


    data = np.concatenate(data_list, axis=-1)

    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def process_and_save_sets(sets, args):
    """
    Process each set by calling 'generate_graph_seq2seq_io_data' and save the
    generated NPZ files.
    """
    for i, dataset in enumerate(sets):
        set_dir = os.path.join(args.output_dir, f'set_{i + 1}')
        os.makedirs(set_dir, exist_ok=True)

        x_offsets = np.sort(
            # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
            np.concatenate((np.arange(-11, 1, 1),))
        )
        # Predict the next one hour
        y_offsets = np.sort(np.arange(1, 13, 1))

        x, y = generate_graph_seq2seq_io_data(
            dataset,
            x_offsets=x_offsets,
            y_offsets=y_offsets,
            add_time_in_day=True,
            add_day_in_week=True,
        )

        print("x shape: ", x.shape, ", y shape: ", y.shape)
        num_samples = x.shape[0]
        num_test = round(num_samples * 0.2)
        num_train = round(num_samples * 0.7)
        num_val = num_samples - num_test - num_train

        x_train, y_train = x[:num_train], y[:num_train]
        x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
        x_test, y_test = x[-num_test:], y[-num_test:]

        for cat in ["train", "val", "test"]:
            _x, _y = locals()["x_" + cat], locals()["y_" + cat]
            print(cat, "x: ", _x.shape, "y:", _y.shape)
            np.savez_compressed(
                os.path.join(set_dir, f"{cat}.npz"),
                x=_x,
                y=_y,
                x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
                y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
            )


def main(args):
    df = pd.read_hdf(args.traffic_df_filename)
    sets = split_data(df)
    process_and_save_sets(sets, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="data/PEMS-BAY", help="Output directory."
    )
    parser.add_argument(
        "--traffic_df_filename",
        type=str,
        default="data/pems-bay.h5",
        help="Raw traffic readings.",
    )
    args = parser.parse_args()
    main(args)
