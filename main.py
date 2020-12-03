from utils import *


input_size = prepare_data()

# make_folded_sets()
#
# metrics = classify(input_size)
#
# show_metrics(metrics)

test_selected_model(input_size, hidden=2, density=2.5, activation=tanh, lr=0.01, epochs=20)