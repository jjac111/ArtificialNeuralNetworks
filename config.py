from keras.activations import tanh, sigmoid, hard_sigmoid, relu, softsign
from keras.optimizers import Adam, SGD

features_to_remove = [38, 35, 11, 27, 17, 7, 12, 32, 9, 36]

data_path = 'data/'
preprocessed_data_path = data_path + 'preprocess/'
train_filename = data_path + 'SPECTF.train'
test_filename = data_path + 'SPECTF.test'

ten_fold_data_path = data_path + '10-fold/'

model_configs = [# {'hidden': 1, 'neuron_numerosity': 2, 'activation': tanh},
                 {'hidden': 2, 'neuron_numerosity': 2.5, 'activation': tanh},
                 # {'hidden': 1, 'neuron_numerosity': 2, 'activation': sigmoid},
                 {'hidden': 2, 'neuron_numerosity': 2.5, 'activation': sigmoid},
                 # {'hidden': 1, 'neuron_numerosity': 2, 'activation': hard_sigmoid},
                 {'hidden': 2, 'neuron_numerosity': 2.5, 'activation': hard_sigmoid},
                 {'hidden': 1, 'neuron_numerosity': 2, 'activation': relu},
                 {'hidden': 2, 'neuron_numerosity': 2.5, 'activation': relu},
                 #  {'hidden': 1, 'neuron_numerosity': 2, 'activation': softsign},
                 # {'hidden': 2, 'neuron_numerosity': 2.5, 'activation': softsign}
                 ]

hyperparameters = {'lr': [0.1, 0.01, 0.005], 'epochs': [i/100 for i in [500, 600, 700, 800, 900, 1000]], 'optimizer': [SGD]}
