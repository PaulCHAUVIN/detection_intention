# Imports
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from transformers import pipeline
from sklearn.metrics import accuracy_score, recall_score
import pandas as pd
from utils import get_label_max_score
import argparse

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Add argparse argument for path_to_data_csv and model_chosen
parser = argparse.ArgumentParser()
parser.add_argument("data_path_to_csv", help="path to the csv file")
parser.add_argument("model_chosen", help="model to use for classification")
args = parser.parse_args()


def model_evaluation(data_path_to_csv, model_chosen):
    """
    :param data_path_to_csv: must be a csv with 2 headers, one text, one label.
    :param model_chosen: either 'English' if csv in English. Or 'Multilanguage' if csv is in other languages. Can deal with up to 12 languages.
    :return: the accuracy of the model on all the sentences of the csv provided.
    """

    # Choose the model
    if model_chosen == 'English':
        classifier = pipeline("zero-shot-classification")
    elif model_chosen == 'Multilanguage':
        classifier = pipeline("zero-shot-classification", model='joeddav/xlm-roberta-large-xnli')

    # Load Data
    df = pd.read_csv(data_path_to_csv)

    # Define our labels
    labels = ['translate', 'travel_alert', 'flight_status',
              'lost_luggage', 'travel_suggestion', 'carry_on',
              'book_hotel', 'book_flight', 'out_of_scope']

    # predict the label based on the maximum probability label
    predicted_labels = []
    for sentence in df['text']:
        if model_chosen == 'English':
            predicted_labels.append(get_label_max_score(classifier(sentence, labels, multi_label=True)))
        elif model_chosen == 'Multilanguage':
            predicted_labels.append(get_label_max_score(classifier(sentence, labels, multi_label=True)))

    # Calculate accuracy for all our dataset
    label_dict = {label: i for i, label in enumerate(set(df['label']))}

    # Convert the labels to integers
    y = [label_dict[label] for label in df['label']]
    y_predicted = [label_dict[label] for label in predicted_labels]

    # Calculate accuracy based on the given dataset
    accuracy = accuracy_score(y, y_predicted)

    return "Accuracy: ", accuracy


print(model_evaluation(args.data_path_to_csv, args.model_chosen))
