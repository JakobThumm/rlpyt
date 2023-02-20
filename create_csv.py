"""Script to convert multiple training runs to single csv."""

import os
import argparse
import pandas as pd
import numpy as np



def average_all_seeds(path_to_folder):
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

    # loop through each subfolder and load the progress.csv file into a pandas dataframe
    for subfolder in subfolders:
        csv_file = os.path.join(subfolder, 'run_0', 'progress.csv')
        df = pd.read_csv(csv_file)
        reward_arrays.append(df['ReturnAverage'].values)
        cost_arrays.append(df['CostAverage'].values)

    # concatenate the numpy arrays on a new axis
    reward_array = np.stack(reward_arrays, axis=0)
    cost_array = np.stack(cost_arrays, axis=0)

    # calculate the mean and standard deviation over all seeds
    reward_mean = np.mean(reward_array, axis=0)
    reward_std = np.std(reward_array, axis=0)
    cost_mean = np.mean(cost_array, axis=0)
    cost_std = np.std(cost_array, axis=0)

    # create a new dataframe with the calculated values
    output_df = pd.DataFrame({
        'step': df['Diagnostics/CumSteps'].values,
        'reward_mean': reward_mean,
        'reward_std': reward_std,
        'cost_mean': cost_mean,
        'cost_std': cost_std
    })

    # output the dataframe to a csv file
    output_df.to_csv(os.path.join(path_to_folder, 'summary.csv'), index=False)


def main():
    """Parse command line arguments and call the csv function."""
    # create a parser for command line arguments
    parser = argparse.ArgumentParser(description='Calculate mean and standard deviation of progress.csv files over all seeds.')
    parser.add_argument('path', type=str, help='Path to folder containing progress.csv files.')

    # parse the command line arguments
    args = parser.parse_args()

    # call the average_all_seeds function with the parsed arguments
    average_all_seeds(args.path)

if __name__ == '__main__':
    main()
