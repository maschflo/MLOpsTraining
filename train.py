import logging
import os
import mlflow
import pandas as pd
import argparse
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, \
    ConfusionMatrixDisplay

def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "md5").hexdigest()

def load_data(path):
    data = pd.read_csv(path)
    return data


def prepare_data(data, random_state=22):
    features = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]",
                "Tool wear [min]", "Type_L", "Type_M", "Type_H"]
    y = data["Machine failure"]
    X = data[features]
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)
    return train_X, test_X, train_y, test_y


def train_model(X, y, random_state=22, max_depth=None, n_estimators=100, class_weight:None|str="balanced"):
    model = RandomForestClassifier(class_weight=class_weight, random_state=random_state, max_depth=max_depth,
                                   n_estimators=n_estimators)
    model.fit(X, y)
    return model


def verify_model(model, X, y):
    val_predictions = model.predict(X)
    conf_mat = confusion_matrix(y, val_predictions)
    class_rep = classification_report(y, val_predictions)
    return val_predictions, conf_mat, class_rep


def predict_with_threshold(model, X, threshold=0.5):
    proba = model.predict_proba(X)[:, 1]
    y_pred_t = (proba >= threshold).astype(int)  # True/False → 1/0
    return y_pred_t


def get_metrics(truth, pred):
    return precision_score(truth, pred), recall_score(truth, pred), f1_score(truth, pred)


def get_confusion_matrix_img(truth, pred):
    return ConfusionMatrixDisplay.from_predictions(truth, pred).figure_


def main():
    if "MLFLOW_TRACKING_URI" not in os.environ:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="./data/test_data.csv")
    parser.add_argument("-t", "--threshold", type=float, default=0.5)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=22)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--class-weight", type=str, default="balanced")
    args = parser.parse_args()
    class_weight = None if args.class_weight == "None" else args.class_weight
    mlflow.set_experiment("ai4i-failure-1")
    with mlflow.start_run() as run:
        mlflow.log_params(
            {"random_state": args.random_seed,
             "threshold": args.threshold,
             "data_hash": file_hash(args.data_path)})
        data = load_data(args.data_path)
        logging.info("Data loaded")
        train_X, test_X, train_y, test_y = prepare_data(data, random_state=args.random_seed)
        logging.info("Data prepared")
        model = train_model(train_X, train_y, random_state=args.random_seed, n_estimators=args.n_estimators,
                            max_depth=args.max_depth, class_weight=class_weight)
        mlflow.log_params(model.get_params())
        logging.info("Model trained")
        pred = predict_with_threshold(model, test_X, threshold=args.threshold)
        logging.info("Model tested")
        precision, recall, f1 = get_metrics(test_y, pred)
        mlflow.log_metrics({"test_precision": precision, "test_recall": recall, "test_f1": f1})
        mlflow.log_figure(get_confusion_matrix_img(test_y, pred), "confusion_matrix.png")
        mlflow.sklearn.log_model(model, name="model", input_example=train_X.head(), skops_trusted_types=["sklearn.tree._tree.Tree"])
        logging.info("Run finished.")
        logging.info("Experiment-ID: %s", run.info.experiment_id)
        logging.info("Run-ID: %s", run.info.run_id)



if __name__ == "__main__":
    main()
