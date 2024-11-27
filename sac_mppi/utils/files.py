#!/usr/bin/env python3
#
# Created on Thu Jun 29 2023 18:56:15
# Author: Mukai (Tom Notch) Yu, Yao He
# Email: mukaiy@andrew.cmu.edu, yaohe@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute, the AirLab
#
# Copyright Ⓒ 2023 Mukai (Tom Notch) Yu, Yao He
#
import csv
import json
import os
import warnings

import cv2
import magic
import numpy as np
import toml
import yaml


mime = magic.Magic(mime=True, uncompress=True)

file_cache = {}  # cache for files that are read
# Map OpenCV data types to NumPy data types
dtype_map = {"u": np.uint8, "i": np.int32, "f": np.float32, "d": np.float64}


def opencv_matrix_constructor(loader, node):
    """Custom constructor for !!opencv-matrix tag."""

    # Parse the node as a dictionary
    matrix_data = loader.construct_mapping(node, deep=True)

    # Extract rows, cols, dt, and data
    rows = matrix_data["rows"]
    cols = matrix_data["cols"]
    dt = matrix_data["dt"]
    data = matrix_data["data"]

    # Determine the NumPy data type
    dtype = dtype_map.get(dt, np.float64)

    # Convert data to a NumPy array and reshape
    matrix = np.array(data, dtype=dtype).reshape((rows, cols))

    return matrix


yaml.add_constructor("tag:yaml.org,2002:opencv-matrix", opencv_matrix_constructor)


def print_dict(d: dict, indent: int = 0) -> None:
    """print a dictionary with indentation, uses recursion to print nested dictionaries

    Args:
        d (dict): dictionary to be printed
        indent (int, optional): indentation level. Defaults to 0.

    Returns:
        None
    """
    for key, value in d.items():
        if isinstance(value, dict):
            print("  " * indent + str(key) + ": ")
            print_dict(value, indent + 1)
        elif isinstance(value, np.ndarray):
            print(
                "  " * indent + str(key) + ":\n" + str(value)
            )  # new line before printing matrix for better readability
        else:
            print("  " * indent + str(key) + ": " + str(value))


def parse_path(probe_path: str, base_path: str = None):
    """parse a potential path, expand ~ to user home, and check if the path exists

    Args:
        probe_path (str): the path to be parsed
        base_path (str, optional): the base path of the current (yaml) file. Defaults to None.

    Returns:
        the parsed path if it exists, False otherwise
    """
    expand_path = os.path.expanduser(probe_path)  # expand ~ to usr home
    if base_path is None:
        base_path = os.getcwd()
    if os.path.isabs(expand_path) and os.path.exists(expand_path):
        return os.path.realpath(expand_path)
    elif os.path.exists(os.path.join(base_path, probe_path)):
        return os.path.realpath(os.path.join(base_path, probe_path))
    else:
        return False


def parse_content(node, yaml_base_path: str):
    """recursively look into the leaf node of a yaml file, if the leaf node is a string, try to parse it as a path and read the file, could be an image or nested yaml config

    Args:
        node: file node to be parsed
        yaml_base_path (str): the base path of the current yaml file

    Returns:
        the read content
    """

    if isinstance(node, dict):
        for key, value in node.items():
            node[key] = parse_content(value, yaml_base_path)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            node[index] = parse_content(value, yaml_base_path)
    elif isinstance(node, str):
        path = parse_path(node, yaml_base_path)
        if path:
            node = read_file(path)

    return node


def read_file(path: str):
    """test the path, read a file.
       supports multiple file type and interleaving config file types.

    Args:
        path (str): path to the file, can be absolute or relative

    Returns:
        content of the file
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist")

    if os.path.isdir(path):
        return path

    if path in file_cache.keys():
        return file_cache[path]

    # determine file type with python-magic, useful for massive image types
    mime_type = mime.from_file(path)

    file_base_path = os.path.dirname(path)

    if mime_type in {
        "application/yaml",
        "application/x-yaml",
        "text/yaml",
    } or path.endswith((".yaml", ".yml")):
        with open(path, "r") as f:
            parsed_yaml = yaml.load(f.read(), Loader=yaml.FullLoader)
        parsed_content = parse_content(parsed_yaml, file_base_path)
    elif mime_type.startswith("image/"):
        parsed_content = cv2.imread(path)
    elif mime_type in {"text/csv", "application/csv"} or path.endswith(".csv"):
        with open(path, "r") as f:
            reader = csv.reader(f)
            parsed_content = list(reader)
    elif mime_type in {"application/json", "text/json"} or path.endswith(".json"):
        with open(path, "r") as f:
            parsed_content = parse_content(json.load(f), file_base_path)
    elif mime_type in {"application/toml", "text/toml"} or path.endswith(".toml"):
        parsed_content = parse_content(toml.load(path), file_base_path)
    elif path.endswith((".npy", ".npz")):
        try:
            parsed_content = np.load(path, allow_pickle=True)
        except Exception as e:
            raise RuntimeError(f"Error loading NumPy array from {path}: {e}")
    # elif mime_type in {'application/x-torchscript', 'application/x-tensorrt'} or path.endswith((".ts", ".trt")):
    #     parsed_content = torch.jit.load(model_path)
    else:
        warnings.warn(f"File {path} is not supported, reading as string")
        parsed_content = path

    file_cache[path] = parsed_content
    return parsed_content
