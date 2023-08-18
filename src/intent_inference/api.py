import time
from typing import List, Any
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


def compute_predictions(
    df: pd.DataFrame,
    selections: List[Any],
    dimensions: List[str],
    row_id_label = "index",
    n_top_predictions=10,
):
    """
    Args:
        df: Dataframe on which predictions are to be made
        dimensions: List of dimensions to predict over
        selections: List of selections

    Returns: List of predictions

    Compute predictions for a given dataframe, dimensions, and selections.
    Returns a list of predictions.
    """
    predictions = []

    if row_id_label not in df:
        df = df.reset_index(names=row_id_label) # if no label column, use index 

    intents = compute_intents(df, dimensions) # compute all algorithm outputs


    for intent in intents:
        predictions.extend(Prediction.from_intent(intent, df, selections, row_id_label)) # compare to user selections


    high_ranking_preds = list(filter(lambda x: x.rank_jaccard > 0.5, predictions)) # predictions with more than 0.5 jaccard_similarity
    high_ranking_preds = sort_and_keep_unique(high_ranking_preds)

    if len(high_ranking_preds) < n_top_predictions: # if the list is empty 
        predictions = sort_and_keep_unique(predictions) # sort and keep unique entries (based on combo of intent, algo & jaccard_similarity)
        predictions = predictions[:n_top_predictions] # take top n preds
    else: 
        predictions = high_ranking_preds 

    return predictions


def run_predictions(df: pd.DataFrame, dimensions: List[str], selections: List[any], row_id_label = "index", n_top_predictions=10):
    """
    Compute predictions for a given dataframe, dimensions, and selections.
    Returns a list of predictions as well as the time taken to generate them.
    """
    start_time = time.time()

    preds = compute_predictions(df, selections, dimensions, row_id_label, n_top_predictions)

    end_time = time.time() - start_time

    ret = {"predictions": preds, "time": end_time}

    return ret


def apply_prediction(
    df: pd.DataFrame, prediction: Prediction, row_id_label = "index"
):
    """
    Apply a given prediction to a dataframe.
    Returns a new list of predictions.
    """

    if row_id_label not in df:
        df = df.reset_index(names=row_id_label) # if no label column, use index 

    # Using intent.apply
    intent = Intent(
        prediction["intent"],
        prediction["algorithm"],
        prediction["dimensions"],
        prediction["params"],
        prediction["info"],
    )

    new_ids = intent.apply(df, row_id_label)

    # update to return a better data structure
    return {
        "ids": new_ids,
        "prediction": Prediction.from_intent(intent, df, new_ids, row_id_label),
    }


def apply_and_generate_predictions(df: pd.DataFrame, prediction: Prediction):
    """
    Apply a given prediction to a dataframe.
    Returns a new list of predictions.
    """
    sel = apply_prediction(df, prediction)

    return compute_predictions(df, prediction["dimensions"], sel)
