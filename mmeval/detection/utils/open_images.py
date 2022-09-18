# Copyright (c) OpenMMLab. All rights reserved.

import json
import csv
import numpy as np


def _convert_hierarchy_tree(hierarchy_map: dict,
                            label_index_map: dict,
                            relation_matrix: np.ndarray,
                            parents: list = [],
                            get_all_parents: bool = True) -> np.ndarray:
    """Get matrix of the corresponding relationship between the parent
    class and the child class.

    Args:
        hierarchy_map (dict): Including label name and corresponding
            subcategory. Keys of dicts are:
            - `LabeName` (str): Name of the label.
            - `Subcategory` (dict | list): Corresponding subcategory(ies).
        label_index_map (dict): The mapping of label name to description names.
        relation_matrix (ndarray): The matrix of the corresponding
            relationship between the parent class and the child class,
            of shape (class_num, class_num).
        parents (list): Corresponding parent class.
        get_all_parents (bool): Whether get all parent names.
            Default: True

    Returns:
        ndarray: The matrix of the corresponding relationship between the 
        parent class and the child class, of shape (class_num, class_num).
    """
    if 'Subcategory' not in hierarchy_map:
        return relation_matrix
    
    for node in hierarchy_map['Subcategory']:
        if 'LabelName' not in node:
            continue

        children_name = node['LabelName']
        children_index = label_index_map[children_name]
        children = [children_index]

        if len(parents) > 0:
            for parent_index in parents:
                if get_all_parents:
                    children.append(parent_index)
                relation_matrix[children_index, parent_index] = 1
        relation_matrix = _convert_hierarchy_tree(
            node, label_index_map, relation_matrix, parents=children)
    return relation_matrix


def get_relation_matrix(hierarchy_file: str, label_file: str) -> np.ndarray:
    """Get the matrix of class hierarchy from the hierarchy file. Hierarchy
    for 600 classes can be found at https://storage.googleapis.com/openimages/
    2018_04/bbox_labels_600_hierarchy_visualizer/circle.html.

    Reference: https://github.com/open-mmlab/mmdetection/blob/
    9d3e162459590eee4cfc891218dfbb5878378842/mmdet/datasets/openimages.py#L357

    Args:
        hierarchy_file (str): File path to the hierarchy for classes.
            E.g. https://storage.googleapis.com/openimages/2018_04/
            bbox_labels_600_hierarchy.json
        label_file (str): File path to the mapping of label name to class
            description names. E.g. https://storage.googleapis.com/openimages
            /2018_04/class-descriptions-boxable.csv

    Returns:
        np.ndarray: The matrix of the corresponding relationship between
        the parent class and the child class, of shape (class_num, class_num).
    """
    index_list = []
    classes_names = []
    with open(label_file, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            classes_names.append(line[1])
            index_list.append(line[0])
    label_index_map = {index: i for i, index in enumerate(index_list)}

    with open(hierarchy_file) as f:
        hierarchy_map = json.load(f)

    class_num = len(label_index_map)
    relation_matrix = np.eye(class_num, class_num)
    relation_matrix = _convert_hierarchy_tree(hierarchy_map,
                                              label_index_map,
                                              relation_matrix)
    return relation_matrix
