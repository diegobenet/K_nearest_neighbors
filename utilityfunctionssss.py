""" utilityfunctions.py
    This python script has the functions needed to implement the K-Nearest Neighbours classification.

    Authors:
        Diego Elizondo Benet                567003
        Alejandro Flores Ramones            537489
        Karla Lira Rangel                   526389
    Emails:
        diego.elizondob@udem.edu
        alejandro.floresr@udem.edu
        karla.lira@udem.edu

    Institution: Universidad de Monterrey
    First created: Wednesday  11 Nov 2020

    We hereby declare that we've worked on this activity with academic integrity.
"""

# import standard libraries
import numpy as np
import pandas as pd
import math


def load_data(path_and_filename, flag):
    """"
    This function reads the data of an external file and uses it to initialize the training and testing data, for it to
    calculate some statistics like mean, media, std, min and max. At last it prints the results and converts it to a
    numpy-type matrix

    INPUTS:
        path_and_filename: String representing the name and location of the file.
        flag: Number representing if the randomized data should be seeded or not.

    OUTPUTS:
        data_training: numpy-type matrix with the attributes representing each feature of training.
        mean_training: numpy-type vector with the mean of each feature of training.
        std_training: numpy-type vector with the standard deviation of training.
        data_testing: numpy-type matrix with the attributes representing each feature of testing.
        mean_testing: numpy-type vector with the mean of each feature of testing.
        std_testing: numpy-type vector with the standard deviation of testing.
    """

    pd.set_option('display.max_rows', 8)
    pd.set_option('display.max_columns', 7)
    pd.set_option('display.width', 1000)

    data_training = []
    mean_training = []
    std_training = []
    min_training = []
    max_training = []
    median_training = []

    data_testing = []
    mean_testing = []
    std_testing = []
    min_testing = []
    max_testing = []
    median_testing = []

    try:
        # read the file
        data = pd.read_csv(path_and_filename)

        # if flag is 1 use the seed '134' (used for the technical report)
        if flag == 1:
            seed = 134
            np.random.seed(seed)

        # shuffle data
        data = data.iloc[np.random.permutation(len(data))]

        # use 95% for training and 5% for testing
        data_training, data_testing = np.split(data, [int(.95 * len(data))])

        # compute statistics
        mean_training = np.mean(data_training.to_numpy()[:, :-1], axis=0)
        std_training = np.std(data_training.to_numpy()[:, :-1], axis=0)
        min_training = np.min(data_training.to_numpy()[:, :-1], axis=0)
        max_training = np.max(data_training.to_numpy()[:, :-1], axis=0)
        median_training = np.median(data_training.to_numpy()[:, :-1], axis=0)

        mean_testing = np.mean(data_testing.to_numpy()[:, :-1], axis=0)
        std_testing = np.std(data_testing.to_numpy()[:, :-1], axis=0)
        min_testing = np.min(data_testing.to_numpy()[:, :-1], axis=0)
        max_testing = np.max(data_testing.to_numpy()[:, :-1], axis=0)
        median_testing = np.median(data_testing.to_numpy()[:, :-1], axis=0)

    except IOError as e:
        print(e)
        exit(1)

    # print statistics, mean, max, min, deviation and testings data and training data
    print('-' * 90)
    print('Training data and Y (target) outputs')
    print('-' * 90)
    print(data_training)
    print('-' * 90)
    print('Training Statistics')
    print('-' * 90)
    print('Training mean:\n', mean_training)
    print('\nTraining standard deviation:\n', std_training)
    print('\nTraining max:\n', max_training)
    print('\nTraining min:\n', min_training)
    print('\nTraining median:\n', median_training)
    print('-' * 90)
    print('Testing points (features)')
    print('-' * 90)
    print(data_testing)
    print('-' * 90)
    print('Testing Statistics')
    print('-' * 90)
    print('Testing mean:\n', mean_testing)
    print('\nTesting standard deviation:\n', std_testing)
    print('\nTraining max:\n', max_testing)
    print('\nTraining min:\n', min_testing)
    print('\nTraining median:\n', median_testing)

    return data_training.to_numpy(), mean_training, std_training, data_testing.to_numpy(), mean_testing, std_testing, \
           data_testing


def visualize_random(title, data, flag):
    """"
    This function prints 10 randomly selected samples from the data-set on the command-line.

    INPUTS:
        title: string representing the header for visual purposes.
        data: numpy-type matrix with the attributes representing each feature.
        flag: Number representing if the randomized data should be seeded or not.

    OUTPUTS:
        none
    """

    print('-' * 90)
    print(title)
    print('-' * 90)
    num_rows = data.shape[0] - 1

    # if flag is 1 use the seed '23'
    if flag == 1:
        seed = 23
        np.random.seed(seed)
    rand = np.random.random_integers(0, num_rows, 10)

    np.set_printoptions(suppress=True)
    for i in range(len(rand)):
        print(data[rand[i], :])


def normalise_data(title, x, mean, std, flag):
    """"
    This function implements feature scaling on the attributes of each feature of the data-set provided and prints it
    on the command line.

    INPUTS:
        title: string representing the header for visual purposes.
        x: numpy-type matrix with the attributes representing each feature.
        mean: numpy-type vector with the mean of each feature.
        std: numpy-type vector with the standard deviation.
        flag: Number representing if the randomized data should be seeded or not.

    OUTPUTS:
        x_scaled: numpy-type matrix with the attributes representing the scaled data.
    """

    num_cols = x.shape[1]
    x_scaled = np.zeros_like(x)

    # scale each feature
    for i in range(num_cols):
        scale = (x[:, i] - mean[i]) / (math.sqrt((std[i] ** 2) + (10 ** -8)))
        x_scaled[:, i] = scale
    np.set_printoptions(precision=3)
    print('-' * 90)
    print(title)
    print('-' * 90)

    num_rows = x_scaled.shape[0]
    if flag == 1:
        seed = 34
        np.random.seed(seed)
    rand = np.random.random_integers(0, num_rows, 10)

    # prints 10 random normalised data
    for i in range(len(rand)):
        print(x_scaled[rand[i], :])
    return x_scaled


def compute_euclidean_distance(x_training, x_testing):
    """"
    This function computes the euclidean distance for each of the training data on every testing point and prints it
    on the command line.

    INPUTS:
        x_training: numpy-type matrix.
        x_testing: numpy-type matrix.

    OUTPUTS:
        e_distance: numpy-type matrix.
    """

    print('-' * 90)
    print('Euclidean distance')
    print('-' * 90)
    num_rows, num_cols = x_testing.shape
    num_rows_training = x_training.shape[0]

    e_distance = np.zeros(shape=(num_rows, num_rows_training), dtype=float)

    # generates the value of the euclidean distance for every row on the testing data.
    for i in range(num_rows):
        e_distance[i] = (np.sqrt(((x_training - x_testing[i]) ** 2).sum(axis=1)))

    # prints the computed distances on the command-line
    print(e_distance)

    return e_distance


def compute_conditional_probabilities(e_distance, k, y_training, panda_testing):
    """"
    This function computes the conditional probabilities that each testing point has to belong on one class and
    calculates the distance of each class. Finally it then prints the results on the command-line.

    INPUTS:
        e_distance: numpy-type matrix.
        k: number represeting 'k' value.
        y_training : numpy-type matrix.
        panda_testing: panda-type dataframe.

    OUTPUTS:
        probabilty: numpy-type matrix.
        distance: numpy-type matrix.
    """

    print('-' * 90)
    print('Testing point (features)')
    print('-' * 90)
    num_rows = e_distance.shape[0]
    e_distance_sorted = np.sort(e_distance, axis=1)[:, :k]
    e_index = np.argsort(e_distance, axis=1)

    probability = np.zeros(shape=(num_rows, 2), dtype=float)
    distance = np.zeros(shape=(num_rows, 2), dtype=float)

    # counts the positives and negatives of the selected K's of one testing sample and the distance of each.
    for i in range(num_rows):
        contPositive = 0
        distPositive = 0
        contNegative = 0
        distNegative = 0
        for j in range(k):
            if y_training[e_index[i, j]] == 1:
                contPositive = contPositive + 1
                distPositive = distPositive + e_distance_sorted[i, j]
            elif y_training[e_index[i, j]] == 0:
                contNegative = contNegative + 1
                distNegative = distNegative + e_distance_sorted[i, j]
        probability[i, 0] = contPositive / k
        probability[i, 1] = contNegative / k
        distance[i, 0] = distPositive
        distance[i, 1] = distNegative

    del panda_testing['Outcome']
    panda_testing['Prob. Diabetes'] = probability[:, 0]
    panda_testing['Prob. No Diabetes'] = probability[:, 1]
    print(panda_testing.round(decimals=2))

    return probability, distance


def predict(probability, distance):
    """"
    This function predicts wheter a class belongs to one class or the other with the probability and distance previously
    computed.

    INPUTS:
        probability: numpy-type matrix.
        distance: numpy-type matrix.

    OUTPUTS:
        predicted: numpy-type vector.
    """

    num_rows, num_cols = probability.shape
    predicted = np.zeros(shape=num_rows, dtype=float)

    for i in range(num_rows):
        if probability[i, 0] > probability[i, 1]:
            predicted[i] = 1
        elif probability[i, 0] < probability[i, 1]:
            predicted[i] = 0
        elif probability[i, 0] == probability[i, 1]:
            if distance[i, 0] > distance[i, 1]:
                predicted[i] = 0
            elif distance[i, 0] <= distance[i, 1]:
                predicted[i] = 1

    return predicted


def confusion_matrix(predicted, y):
    """
    This function computes the confusion matrix using the predictions previously obtained.
    It then calculates some metrics like accuracy, precision, recall, sensitivity and F1-score.
    At last it calls some methods to print both the confusion matrix and the obtained performance metrics.

    INPUTS:
        predicted: numpy-type vector
        y: numpy-type vector

    OUTPUTS
        none
    """

    num_rows = predicted.shape[0]

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    # build the confusion matrix
    for i in range(num_rows):
        if predicted[i] == 1 and y[i] == 1:
            true_positives += 1
        elif predicted[i] == 1 and y[i] == 0:
            false_positives += 1
        elif predicted[i] == 0 and y[i] == 0:
            true_negatives += 1
        elif predicted[i] == 0 and y[i] == 1:
            false_negatives += 1

    total = true_positives + false_positives + true_negatives + false_negatives

    # compute metrics from the confusion matrix
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)
    f1_score = 2 * ((precision * recall) / (precision + recall))

    # print the confusion matrix
    print_confusion_matrix(true_positives, false_positives, true_negatives, false_negatives)

    # print the confusion matrix metrics
    print_performance_metrics(accuracy, precision, recall, specificity, f1_score, total)


def print_confusion_matrix(true_positives, false_positives, true_negatives, false_negatives):
    """
    This function prints the confusion matrix on the command-line with the corresponding data previously obtained
    on the confusion_matrix method.

    INPUTS:
        true_positives: number representing the total True Positives computed for the confusion matrix
        false_positives: number representing the total False Positives computed for the confusion matrix
        true_negatives: number representing the total True Negatives computed for the confusion matrix
        false_negatives: number representing the total False Negatives computed for the confusion matrix

    OUTPUTS
        none
    """

    print('-' * 90)
    print('Confusion Matrix')
    print('-' * 90)

    space = '\t\t\t\t'
    print(space, '                     Predicted class')
    print(space, '                      1           0')
    print(space, '                 ┌----------┬----------┐')
    print(space, '               1 |    TP    |    FN    |')
    print(space, '   Actual        |   ', true_positives, '\t |   ', false_negatives, '  \t|   ')
    print(space, '   class         ├----------┼----------┤')
    print(space, '               0 |    FP    |    TN    |')
    print(space, '                 |   ', false_positives, '\t |   ', true_negatives, '\t|   ')
    print(space, '                 └----------┴----------┘')
    print('Legend: TP = True Positive\tFP = False Positive\tFN = False Negative\tTN = True Negative')


def print_performance_metrics(accuracy, precision, recall, specificity, f1_score, total):
    """
    This function prints the performance metrics on the command-line with the corresponding data previously obtained
    on the confusion_matrix method.

    INPUTS:
        accuracy: number representing the computed accuracy of the confusion matrix
        precision: number representing the computed precision of the confusion matrix
        recall: number representing the computed recall of the confusion matrix
        specificity: number representing the computed specificity of the confusion matrix
        f1_score: number representing the computed f1_score of the confusion matrix
        total: number representing the total data of the confusion matrix

    OUTPUTS
        none
    """

    print('-' * 90)
    print('Performance metrics')
    print('-' * 90)
    print('Accuracy: \t\t', round(accuracy, 2))
    print('Precision: \t\t', round(precision, 2))
    print('Recall: \t\t', round(recall, 2))
    print('Specificity: \t', round(specificity, 2))
    print('F1 Score: \t\t', round(f1_score, 2))

    print('\nTotal data: ', total)


