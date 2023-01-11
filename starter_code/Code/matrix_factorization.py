from matplotlib import pyplot as plt

from Project.starter_code.utils import *
from scipy.linalg import sqrtm

import numpy as np


def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    # Randomly select a pair (user_id, question_id).
    i = np.random.choice(len(train_data["question_id"]), 1)[0]
    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]
    u_i = u[n]
    z_i = z[q]
    R_ij = c
    # Final partial derivatives
    u_i_T = np.transpose(u_i)
    diff = R_ij - np.matmul(u_i_T, z_i)
    d_u_i = diff * z_i
    d_z_i = diff * u_i
    # Update
    u[n] = u_i + lr * d_u_i
    z[q] = z_i + lr * d_z_i
    return u, z


def als(train_data, val_data, k, lr, num_iteration):
    """ Performs ALS algorithm, here we use the iterative solution - SGD 
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: validation data
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))
    train_losses = []
    valid_losses = []
    for i in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z)
        if i % 100 == 0:
            train_loss = squared_error_loss(train_data, u, z)
            train_losses.append(train_loss)
            valid_loss = squared_error_loss(val_data, u, z)
            valid_losses.append(valid_loss)
    zT = np.transpose(z)
    mat = np.matmul(u, zT)
    return mat, train_losses, valid_losses


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    options = [15, 30, 40, 55, 70]
    optimum = 0
    best_k = 0
    for k in options:
        train_k = svd_reconstruct(train_matrix, k)
        accuracy = sparse_matrix_evaluate(val_data, train_k)
        if accuracy > optimum:
            optimum = accuracy
            best_k = k
    test_accuracy = sparse_matrix_evaluate(test_data, svd_reconstruct(train_matrix, best_k))
    print('The best k is {k}, with a validation accuracy of {accuracy} and a test loss'
          ' of {test}'.format(k=best_k,
                              accuracy=optimum,
                              test=test_accuracy))
    optimum2 = 0
    best_k2 = 0
    for k in options:
        reconstruct, train_losses, valid_losses = als(train_data, val_data, k, 0.1, 20000)
        accuracy = sparse_matrix_evaluate(val_data, reconstruct)
        if accuracy > optimum2:
            optimum2 = accuracy
            best_k2 = k
    iterations = [5000, 10000, 20000]
    learning_rate = [0.1, 0.2, 0.3]
    best_learning_rate = 0
    best_iterations = 0
    best_accuracy = 0
    for a in learning_rate:
        for iteration in iterations:
            reconstruct_train, train_losses, valid_losses = als(train_data, val_data, best_k2, a, iteration)
            accuracy = sparse_matrix_evaluate(val_data, reconstruct_train)
            if accuracy > best_accuracy:
                best_learning_rate = a
                best_iterations = iteration
    print('The best learning rate and iterations are {a}, {b}'.format(a=best_learning_rate, b=best_iterations))
    reconstruct_train, train_losses, valid_losses = als(train_data, val_data, best_k2,
                                                        best_learning_rate, best_iterations)
    test_acc2 = sparse_matrix_evaluate(test_data, reconstruct_train)
    plt.plot(train_losses)
    plt.plot(valid_losses)
    plt.ylabel('Loss')
    plt.xlabel('number of iterations')
    plt.title('number of iterations vs Loss plot for train data')
    plt.show()
    print('The best k is {k}, with a validation accuracy of {accuracy} and '
          'a test accuracy of {test}'.format(k=best_k2,
                                             accuracy=optimum2,
                                             test=test_acc2))


if __name__ == "__main__":
    main()
