""" k_nearest_neighbours.py
    This python script implements the K-Nearest Neighbours classification with the functions provided by
    utilityfunctions.py.

    Authors:
        Diego Elizondo Benet                567003
        Alejandro Flores Ramones            537489
        Karla Lira Rangel                   526389
    Emails:
        diego.elizondob@udem.edu
        alejandro.floresr@udem.edu
        karla.lira@udem.edu

    Institution: Universidad de Monterrey
    First created: Wednesday  13 Dic 2020

    We hereby declare that we've worked on this activity with academic integrity.
"""

# import user-defined libraries
import utilityfunctionssss as uf


def main():
    """"
    This main function calls all the functions needed to get the data and to implement the K-Nearest Neighbours
    classification with said data, as well as to build the confusion matrix and compute some performance metrics.

    INPUTS:
        none

    OUTPUTS:
        none
    """

    # To evaluate our algorithm, we first choose a value of k of 3, 5, 10, 15 or 20.
    k = 10

    # load training and testing data
    # if flag = 1, the data is randomly generated using a seed, else the data is randomly generated from scratch
    # this is used so the the execution can be the same as the one used for the technical report
    flag = 1
    data_training, mean_training, std_training, \
    data_testing, mean_testing, std_testing, panda_testing = uf.load_data('diabetes.csv', flag)

    # visualize 10 random training samples
    # if flag = 1, the data is randomly generated using a seed, else the data is randomly generated from scratch
    flag = 1
    uf.visualize_random('Visualize 10 Random training samples', data_training, flag)

    # visualize 10 random testing samples
    # if flag = 1, the data is randomly generated using a seed, else the data is randomly generated from scratch
    flag = 1
    uf.visualize_random('Visualize 10 Random testing samples', data_testing, flag)

    # obtain the x values and the y values of the training data
    training_num_rows, num_cols = data_training.shape
    x_training = data_training[:, :-1]
    y_training = data_training[:, num_cols - 1].reshape(1, training_num_rows).T

    # obtain the x values and the y values of the testing data
    testing_num_rows = data_testing.shape[0]
    x_testing = data_testing[:, :-1]
    y_testing = data_testing[:, num_cols - 1].reshape(1, testing_num_rows).T

    # get training data normalized
    # if flag = 1, the data is randomly generated using a seed, else the data is randomly generated from scratch
    # this is used so the the execution can be the same as the one used for the technical report
    flag = 1
    x_training_scaled = uf.normalise_data('Training data scaled', x_training, mean_training, std_training, flag)

    # get Testing data normalized
    # if flag = 1, the data is randomly generated using a seed, else the data is randomly generated from scratch
    # this is used so the the execution can be the same as the one used for the technical report
    flag = 1
    x_testing_scaled = uf.normalise_data('Testing data scaled', x_testing, mean_testing, std_testing, flag)

    # Compute the euclidean distance
    e_distance = uf.compute_euclidean_distance(x_training_scaled, x_testing_scaled)
    # Compute the probabilities for the testing samples to belong in one class or the other
    probability, distance = uf.compute_conditional_probabilities(e_distance, k, y_training, panda_testing)

    # Predicts the probabilities for the testing point to belong to one class or the other
    testing_predicted = uf.predict(probability, distance)

    # Compute the confusion matrix and the performance metrics
    uf.confusion_matrix(testing_predicted, y_testing)


main()
