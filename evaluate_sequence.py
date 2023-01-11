# Imports
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from transformers import pipeline
from utils import get_label_max_score
import argparse

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument("sentence", help="the sentence to classify, must be a string")
parser.add_argument("model_chosen", help="model to use for classification, either 'English' or 'Multilanguage'")
args = parser.parse_args()


def evaluate_unique_sequence(sentence, model_chosen):
    """
    :param sentence: must be a string
    :param model_chosen: either 'English' if csv in English. Or 'Multilanguage' if csv is in other languages. Can deal with up to 12 languages.
    :return: the predicted label for the provided sentence from the model.
    """

    # Choose the model
    if model_chosen == 'English':
        classifier = pipeline("zero-shot-classification")
    elif model_chosen == 'Multilanguage':
        classifier = pipeline("zero-shot-classification", model='joeddav/xlm-roberta-large-xnli')

        # Define our classes
    labels = ['translate', 'travel_alert', 'flight_status',
              'lost_luggage', 'travel_suggestion', 'carry_on',
              'book_hotel', 'book_flight', 'out_of_scope']
    # predict the label based on the maximum probability label
    return get_label_max_score(classifier(sentence, labels, multi_label=True))


# Test our model on specific inputs
print(evaluate_unique_sequence(args.sentence, args.model_chosen))
