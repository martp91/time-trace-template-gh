#!/usr/bin/env python
import argparse
import os


def make_dir_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


parser = argparse.ArgumentParser()

parser.add_argument(
    "-data_directory",
    help="type the path of the directory where data is stored",
    default=None,
)

args = parser.parse_args()

data_directory = args.data_directory

if data_directory is None:
    data_directory = (
        input(
            (
                "Please specify where the data trees are stored"
                "[default=/home/mart/auger/data/time_templates/]:"
            )
        )
        or "/home/mart/auger/data/time_templates/"
    )
make_dir_if_not_exist(data_directory)
make_dir_if_not_exist(os.path.join(data_directory, "binned_df"))
make_dir_if_not_exist(os.path.join(data_directory, "mean_df"))
make_dir_if_not_exist(os.path.join(data_directory, "detector_response"))
make_dir_if_not_exist(os.path.join(data_directory, "trees"))
make_dir_if_not_exist(os.path.join(data_directory, "adst"))
make_dir_if_not_exist(os.path.join(data_directory, "fitted"))

with open("data_directory.txt", "w") as fl:
    fl.write(data_directory)
