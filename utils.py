import os
import pandas as pd
import numpy as np
from datetime import timedelta
from time import time
from IPython.display import display, clear_output
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, \
    precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from keras import Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import Callback
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam, SGD
from statistics import stdev, mean
from config import *


def prepare_data():
    if not os.path.exists(preprocessed_data_path):
        os.mkdir(preprocessed_data_path)

    raw_train = pd.read_csv(train_filename, header=None)
    raw_test = pd.read_csv(test_filename, header=None)

    merged = pd.concat([raw_train, raw_test], axis=0, ignore_index=True).drop(columns=features_to_remove).sample(frac=1)

    input_size = len(merged.columns) - 1

    scaler = MinMaxScaler()

    print('Raw dataset:')
    display(merged)

    merged = pd.DataFrame(scaler.fit(merged).transform(merged))

    print('Normalized dataset:')
    display(merged)

    merged.to_csv(preprocessed_data_path + 'normalized.csv', header=False, index=False)

    return input_size


def make_folded_sets():
    if not os.path.exists(ten_fold_data_path):
        os.mkdir(ten_fold_data_path)

    merged = pd.read_csv(preprocessed_data_path + 'normalized.csv', header=None)

    length = int(len(merged) / 10)  # length of each fold
    folds = []
    for i in range(9):
        folds += [merged.iloc[i * length:(i + 1) * length]]
    folds += [merged.iloc[9 * length:len(merged)]]

    for i, test in enumerate(folds):
        train = merged.drop(index=test.index)

        test.to_csv('/'.join([ten_fold_data_path, f'{i + 1}_test.csv']), index=None)
        train.to_csv('/'.join([ten_fold_data_path, f'{i + 1}_train.csv']), index=None)

    print('Folded sets generated. 10th fold set as example:\nTRAIN')
    display(train)
    print('TEST')
    display(test)


def classify(input_size):
    plot_callback = PlotLearning()
    datasets = [{'train': pd.read_csv(f'{ten_fold_data_path}{i + 1}_train.csv'),
                 'test': pd.read_csv(f'{ten_fold_data_path}{i + 1}_test.csv')} for i in range(10)]
    metrics = {}
    bce = BinaryCrossentropy()

    progress = 0
    tot_models = len(model_configs) * len(hyperparameters['lr']) * len(hyperparameters['epochs']) * len(
        hyperparameters['optimizer']) * 10
    start = time()
    now = time() - start
    print(f'{round(100 * progress / tot_models, 2)}% Time spent: {timedelta(seconds=round(now))}')
    for lr in hyperparameters['lr'][:1]:
        for epochs in hyperparameters['epochs'][:1]:
            for optimizer in hyperparameters['optimizer'][:1]:
                models = create_models(input_size)

                for j, model in enumerate(models):
                    accs = []
                    precs = []
                    recs = []
                    aucs = []
                    losses = []
                    for i, d in enumerate(datasets):
                        y_col = d['train'].columns[0]
                        train_X = d['train'].drop(columns=y_col)
                        train_y = d['train'][y_col]
                        test_X = d['test'].drop(columns=y_col)
                        test_y = d['test'][y_col]
                        assert input_size == len(train_X.columns)

                        model.compile(optimizer(lr), loss='binary_crossentropy',
                                      metrics=['accuracy'])

                        history = model.fit(train_X, train_y, batch_size=5, epochs=epochs, verbose=0)
                        progress += 1
                        clear_output()
                        now = time() - start
                        print(f'{round(100 * progress / tot_models, 2)}% Time spent: {timedelta(seconds=round(now))}')

                        classification = [0 if p < 0.5 else 1 for p in model.predict(test_X)]

                        accs.append(accuracy_score(test_y, classification))
                        precs.append(precision_score(test_y, classification))
                        recs.append(recall_score(test_y, classification))
                        aucs.append(roc_auc_score(test_y, classification))
                        losses.append(float(bce(test_y, classification).numpy()))

                    configuration_name = f'{model.name} LR{lr} E{epochs} {optimizer.__name__}'
                    metrics[configuration_name] = {}
                    metrics[configuration_name]['lr'] = lr
                    metrics[configuration_name]['epochs'] = epochs
                    metrics[configuration_name]['optimizer'] = optimizer.__name__
                    metrics[configuration_name]['hidden layers'] = model_configs[j]['hidden']
                    metrics[configuration_name]['density factor'] = model_configs[j]['neuron_numerosity']
                    metrics[configuration_name]['activation'] = model_configs[j]['activation'].__name__

                    metrics[configuration_name]['accuracy'] = (mean(accs), stdev(accs))
                    metrics[configuration_name]['precision'] = (mean(precs), stdev(precs))
                    metrics[configuration_name]['recall'] = (mean(recs), stdev(recs))
                    metrics[configuration_name]['roc_auc'] = (mean(aucs), stdev(aucs))
                    metrics[configuration_name]['loss'] = (mean(losses), stdev(losses))

    return metrics


def create_models(input_size):
    models = []

    for conf in model_configs:
        inp = Input((input_size,))
        hid = []
        hidden_layers = conf['hidden']
        hidden_factor = conf['neuron_numerosity']
        dens = hidden_factor
        act = conf['activation']
        for i in range(hidden_layers):
            if i == 0:
                layer = Dense(int(input_size * dens), activation=act)(inp)
                layer = Dropout(0.5)(layer)
            else:
                layer = Dense(int(input_size * dens), activation=act)(hid[i - 1])
                layer = Dropout(0.5)(layer)
            hid.append(layer)
            dens /= 2
        out = Dense(1, activation='sigmoid')(hid[-1])

        m = Model(inputs=inp, outputs=out, name=f'{hidden_layers}-{hidden_factor}-{act.__name__}')
        models.append(m)

    return models


def show_metrics(metrics):
    df = pd.DataFrame(
        columns=['Model name', 'Hidden Layers', 'Density Factor', 'Activation', 'Learning Rate', 'Epochs', 'Optimizer',
                 'Mean Accuracy', 'Mean Precision', 'Mean Recall', 'Mean AUC', 'Mean Loss', 'STDev Accuracy',
                 'STDev Precision', 'STDev Recall', 'STDev AUC', 'STDev Loss'])

    for name, met in metrics.items():
        row = {'Model name': name, 'Hidden Layers': met['hidden layers'], 'Density Factor': met['density factor'],
               'Activation': met['activation'], 'Learning Rate': met['lr'], 'Epochs': met['epochs'],
               'Optimizer': met['optimizer'], 'Mean Accuracy': met['accuracy'][0],
               'Mean Precision': met['precision'][0],
               'Mean Recall': met['recall'][0], 'Mean AUC': met['roc_auc'][0], 'Mean Loss': met['loss'][0],
               'STDev Accuracy': met['accuracy'][1], 'STDev Precision': met['precision'][1],
               'STDev Recall': met['recall'][1],
               'STDev AUC': met['roc_auc'][1], 'STDev Loss': met['loss'][1]}
        df = df.append(row, ignore_index=True)

    return pd.pivot_table(df, index=['Hidden Layers', 'Density Factor', 'Activation', 'Learning Rate', 'Epochs', 'Optimizer'])


def test_selected_model(input_size, hidden=2, density=2.5, activation=tanh, lr=0.1, epochs=5, optimizer=SGD):
    plot_callback = PlotLearning()
    datasets = [{'train': pd.read_csv(f'{ten_fold_data_path}{i + 1}_train.csv'),
                 'test': pd.read_csv(f'{ten_fold_data_path}{i + 1}_test.csv')} for i in range(10)]

    predictions = []
    classifications = []
    true_labels = []
    histories = []
    for i, d in enumerate(datasets):
        y_col = d['train'].columns[0]
        train_X = d['train'].drop(columns=y_col)
        train_y = d['train'][y_col]
        test_X = d['test'].drop(columns=y_col)
        test_y = d['test'][y_col]

        dens = density
        inp = Input((input_size,))
        hid = []
        for i in range(hidden):
            if i == 0:
                layer = Dense(int(input_size * dens), activation=activation)(inp)
                layer = Dropout(0.3)(layer)
            else:
                layer = Dense(int(input_size * dens), activation=activation)(hid[i - 1])
                layer = Dropout(0.3)(layer)
            hid.append(layer)
            dens /= 2
        out = Dense(1, activation='sigmoid')(hid[-1])

        model = Model(inputs=inp, outputs=out,
                      name=f'{hidden}-{density}-{activation} LR{lr} E{epochs} {optimizer.__name__}')

        model.compile(optimizer(lr), loss='binary_crossentropy',
                      metrics=['accuracy'])

        class_weights = compute_class_weight('balanced', np.unique(train_y), train_y)
        history = model.fit(train_X, train_y, batch_size=5, epochs=epochs, callbacks=[plot_callback],
                            class_weight=class_weights)

        pred = model.predict(test_X)
        predictions.extend(pred)
        classifications.extend([1 if p > 0.5 else 0 for p in pred])
        true_labels.extend(test_y)
        histories.append(history.history)

    display(pd.DataFrame(confusion_matrix(true_labels, classifications),
                         index=['Actual Neg', 'Actual Pos'],
                         columns=['Predicted Neg', 'Predicted Pos']))
    roc_x, roc_y, _ = roc_curve(true_labels, predictions)
    plt.figure()
    plt.plot(roc_x, roc_y)
    plt.title('ROC Curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()

    print(f'ROC AUC:\t{roc_auc_score(true_labels, predictions)}')

    prec, rec, _ = precision_recall_curve(true_labels, predictions)
    plt.figure()
    plt.plot(prec, rec)
    plt.title('Precision vs. Recall Curve')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.show()

    loss = np.array([h['loss'] for h in histories])
    loss_mean = np.mean(loss, axis=0)
    epochs = list(range(len(loss_mean)))
    plt.plot(epochs, loss_mean)
    plt.title('Mean of loss in 10 fold crossvalidation')
    plt.xlabel('Epochs')
    plt.ylabel('Mean of Loss')
    plt.show()


# Code by user kav, obtained from https://gist.github.com/stared/dfb4dfaf6d9a8501cd1cc8b8cb806d2e
# Used for live plotting the loss
class PlotLearning(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

        clear_output(wait=True)

        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()

        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend(['accuracy', f'epoch: {epoch}'])
        plt.show();
#############################################################
