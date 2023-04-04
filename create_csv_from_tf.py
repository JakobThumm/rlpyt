"""Script to convert multiple training runs to single csv."""

import os
import argparse
import pandas as pd
import numpy as np
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def average_all_seeds(path_to_folder, window_size=1):
    """Convert multiple training runs to single csv.

    Take in a path to a folder containing multiple training run seeds,
    and output a csv file with columns
    step, reward_mean, reward_std, cost_mean, cost_std.
    """
    # create a list of subfolders in the given path
    subfolders = [f.path for f in os.scandir(path_to_folder) if f.is_dir()]

    # create an empty list to store the numpy arrays
    reward_arrays = []
    cost_arrays = []
    step_arrays = []
    tag_base = "rollout/"

    # loop through each subfolder and load the progress.csv file into a pandas dataframe
    for subfolder in subfolders:
        summary_iterator = EventAccumulator(subfolder + '/PPO_1/').Reload()
        reward = pd.DataFrame.from_records(
            summary_iterator.Scalars(tag_base + 'ep_rew_mean'),
            columns=summary_iterator.Scalars(tag_base + 'ep_rew_mean')[0]._fields)["value"]
        cost = pd.DataFrame.from_records(
            summary_iterator.Scalars(tag_base + 'cumulative_cost'),
            columns=summary_iterator.Scalars(tag_base + 'cumulative_cost')[0]._fields)["value"]
        steps = pd.DataFrame.from_records(
            summary_iterator.Scalars(tag_base + 'ep_rew_mean'),
            columns=summary_iterator.Scalars(tag_base + 'ep_rew_mean')[0]._fields)["step"]
        reward_arrays.append(reward.values)
        cost_arrays.append(cost.values)
        step_arrays.append(steps.values)

    # concatenate the numpy arrays on a new axis
    reward_array = np.stack(reward_arrays, axis=0)
    cost_array = np.stack(cost_arrays, axis=0)
    # Make sure the window size is odd
    if window_size % 2 == 0:
        window_size += 1
    half_window = (window_size-1) // 2
    reward_mean = np.zeros(reward_array.shape[1]-2*half_window)
    reward_std = np.zeros(reward_array.shape[1]-2*half_window)
    cost_mean = np.zeros(reward_array.shape[1]-2*half_window)
    cost_std = np.zeros(reward_array.shape[1]-2*half_window)
    step = np.zeros(reward_array.shape[1]-2*half_window)
    for i in range(half_window, reward_array.shape[1]-half_window):
        reward_mean[i-half_window] = np.mean(reward_array[:, i-half_window:i+half_window])
        reward_std[i-half_window] = np.std(reward_array[:, i-half_window:i+half_window])
        cost_mean[i-half_window] = np.mean(cost_array[:, i-half_window:i+half_window])
        cost_std[i-half_window] = np.std(cost_array[:, i-half_window:i+half_window])
        step[i-half_window] = step_arrays[0][i]

    # create a new dataframe with the calculated values
    output_df = pd.DataFrame({
        'step': step,
        'ReturnAverage': reward_mean,
        'ReturnStd': reward_std,
        'CostAverage': cost_mean,
        'CostStd': cost_std
    })
    # output the dataframe to a csv file
    # get name of last folder in path_to_folder
    folder_name = os.path.basename(os.path.normpath(path_to_folder))
    output_df.to_csv(os.path.join(path_to_folder, '{}.csv'.format(folder_name)), index=False)


def main():
    """Parse command line arguments and call the csv function."""
    # create a parser for command line arguments
    parser = argparse.ArgumentParser(description='Calculate mean and standard deviation of progress.csv files over all seeds.')
    parser.add_argument('path', type=str, help='Path to folder containing progress.csv files.')
    parser.add_argument('window_size', type=int, help='Window size for rolling average.')


    # parse the command line arguments
    args = parser.parse_args()

    # call the average_all_seeds function with the parsed arguments
    average_all_seeds(args.path, args.window_size)

if __name__ == '__main__':
    main()
