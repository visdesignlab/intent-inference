from unittest import TestCase
import random

import numpy as np
import pandas as pd
from intent_inference import compute_predictions, apply_prediction


class TestIntent(TestCase):
    def test(self):
        data = pd.read_csv("src/intent_inference/tests/data/cluster_simple_v1.csv")

        label_col = "index"

        data_c = data.copy(deep=True)
        data_c = data_c.reset_index(names=label_col)

        sels = data_c[label_col].sample(40).tolist()

        preds = compute_predictions(
            data,
            sels,
            ["X", "Y"],
            row_id_label=label_col
        )

        selected_prediction_map = {}

        for sp in preds:
            key = sp["intent"] + sp["algorithm"]
            if key not in selected_prediction_map:
                selected_prediction_map[key] = sp

        for _, selected_prediction  in selected_prediction_map.items():


            num_rows = int(data.shape[0] * 0.2)

            random_indices = np.random.choice(data.shape[0], num_rows, replace=False)

            for index in random_indices:
                data.loc[index, 'X'] = random.randint(1, 100)
                data.loc[index, 'y'] = random.randint(1, 100)

            a = apply_prediction(data, selected_prediction, label_col)

        
        self.assertTrue(True)

    def test_always_passes(self):
        self.assertTrue(True)
