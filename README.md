# detection_intention


This project uses zero shot classification models from Hugging Face. 

You can : 
- Predict the label of a given input thanks to the script 'evaluate_sentence.py' by prompting into a terminal a command line such as
"python evaluate_sequence.py "Is there an issue with my flight reservation?" English" for evaluation in english
or "python evaluate_sequence.py "Y a t il un problème avec ma réservation d'avion ?" Multilanguage" if you want to evaluate a sentence to another language

- Evaluate the model performances on several sentences by uploading a csv with two headers text and label. 
Such task can be done by prompting into a command line "python evaluate_csv.py path_to_csv English" for csv in english  
and "python evaluate_csv.py path_to_csv Multilanguage" for csv in another language than english

