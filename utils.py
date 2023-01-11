def get_label_max_score(input_value):
    zipped = zip(input_value['labels'], input_value['scores'])
    if max(input_value['scores']) >= 0.75:
        max_label = max(zipped, key=lambda x: x[1])[0]
    else:
        max_label = 'out_of_scope'
    return max_label

