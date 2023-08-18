from typing import List

import pandas as pd

from ...compute.range import get_decision_paths, range_clf
from .base import AlgorithmBase


class Range(AlgorithmBase):
    def __init__(
        self,
        data: pd.DataFrame,
        dimensions: List[str],
        selections,
        row_id_label,
        max_depth = None
    ):
        self.algorithm = "DT"
        self.intent = "Range"
        self.dimensions = dimensions

        selection_mask = data[row_id_label].isin(selections)

        clf = range_clf(data[dimensions], selection_mask, max_depth=max_depth)

        paths, mask, labels = get_decision_paths(clf, data[dimensions], selection_mask)

        self.labels = labels
        self.params = {
                "max_depth": max_depth
        }
        self.info = {
            "depth": clf.get_depth(),
            "paths": paths[0],
            "mask": mask.tolist(),
            "labels": labels.tolist()
        }

    @staticmethod
    def compute(data: pd.DataFrame, dimensions, selections, row_id_label):
        prev_instance_depth = None

        for _ in range(0, 2):
            if not prev_instance_depth:
                instance = Range(data, dimensions, selections, row_id_label)
                prev_instance_depth = instance.info["depth"]
                yield instance
            elif prev_instance_depth > 1:
                instance = Range(data, dimensions, selections, row_id_label, prev_instance_depth - 1)
                yield instance

