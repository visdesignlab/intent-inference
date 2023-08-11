import time
from typing import List
import pandas as pd

from .inference.intent import Intent
from .inference.inference import compute_intents
from .inference.prediction import Prediction
from .inference.inference import sort_and_keep_unique

# API: user provides dataframe, dimensions, user_selections
# API: returns list of predictions as JSON (map to_dict() over list)

# API: part 2: apply previous prediction to updated dataframe
#              selection is previous in prediction not selected and matches
# API: part 2: apply_prediction(prediction, dataframe)
# API: part 2: returns new list of predictions


def compute_predictions(df: pd.DataFrame, dimensions: List[str], selections: List[any]):
    """
    Compute predictions for a given dataframe, dimensions, and selections.
    Returns a list of predictions.
    """
    predictions = []
    intents = compute_intents(df, dimensions)

    for intent in intents:
        predictions.extend(Prediction.from_intent(intent, df, selections))

    high_ranking_preds = list(filter(lambda x: x.rank_jaccard > 0.5, predictions))

    if len(high_ranking_preds) == 0:
        return []

    sorted_predictions = sort_and_keep_unique(high_ranking_preds)

    if len(high_ranking_preds) >= 10:  # potentially parameterize this value
        predictions = high_ranking_preds
    else:
        predictions = sorted_predictions[:10]

    return predictions


def run_predictions(df: pd.DataFrame, dimensions: List[str], selections: List[any]):
    """
    Compute predictions for a given dataframe, dimensions, and selections.
    Returns a list of predictions as well as the time taken to generate them.
    """
    start_time = time.time()

    preds = compute_predictions(df, dimensions, selections)

    end_time = time.time() - start_time

    json_ret = {"predictions": preds, "time": end_time}

    return json_ret


def apply_prediction(df: pd.DataFrame, prediction: Prediction):
    """
    Apply a given prediction to a dataframe.
    Returns a new dataframe.
    """
    # Using intent.apply
    intent = Intent(
        prediction['intent'],
        prediction['algorithm'],
        prediction['dimensions'],
        prediction['params'],
        prediction['info'])

    new_ids = intent.apply(df)

    print("using intent.apply")

    return compute_predictions(df, intent.dimensions, new_ids)

    # Using ipns and matches to generate new selection from prediction
    # ipns = prediction['membership_stats']['ipns']
    # matches = prediction['membership_stats']['matches']

    # selections = list(set(ipns + matches))

    # print("using ipns and matches")

    # return compute_predictions(df, prediction['dimensions'], selections)
