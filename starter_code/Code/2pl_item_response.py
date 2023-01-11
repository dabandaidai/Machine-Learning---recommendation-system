from scipy.stats import ttest_ind

from item_response import irt
from item_response import evaluate_1pl

from utils import *

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def sigmoid(x):
    """ Apply sigmoid function.
    """
    # e(x) / [1+e(x)] = 1 / [1+e(-x)]
    return np.exp(x) / (1 + np.exp(x))


# def neg_log_likelihood(data, theta, beta, c):
#     """ Compute the negative log-likelihood.
#     :param data: A dictionary {user_id: list, question_id: list,
#     is_correct: list}
#     :param theta: Vector
#     :param beta: Vector
#     :param c: Vector
#     :return: float
#     """
#     log_lklihood = 0
#     user_ids = data["user_id"]
#     question_ids = data["question_id"]
#     c_ijs = data["is_correct"]
#
#     # Precondition: len(user_ids) == len(question_ids) == len(c_ijs)
#     for node in range(len(c_ijs)):
#         theta_i = theta[user_ids[node]]
#         beta_j = beta[question_ids[node]]
#         c_ij = c_ijs[node]
#         c_j = c[question_ids[node]]
#         e = np.exp(theta_i - beta_j)
#         # l(tt_i, bt_j, c_j) = sum[
#         #   c_ij*log(c_j+e) + (1-c_ij)log(1-c_j) - log(1+e) ]
#         log_lklihood += c_ij * np.log(c_j + e) + (1 - c_ij) * np.log(1 - c_j) - np.log(1 + e)
#     return -log_lklihood


def update_theta_beta_c(data, lr, theta, beta, c):
    """ Update theta, beta, and c using gradient descent.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float, represents learning rate
    :param theta: Vector
    :param beta: Vector
    :param c: Vector
    :return: tuple of vectors
    """
    user_ids = data["user_id"]
    question_ids = data["question_id"]
    c_ijs = data["is_correct"]
    theta_per_stud = [[] for i in range(len(theta))]
    beta_per_ques = [[] for i in range(len(beta))]
    c_per_ques = [[] for i in range(len(c))]

    # Precondition:
    #   len(user_ids) == len(question_ids) == len(c_ijs)
    #   user_id in [0, N) and question_id in [0, M)
    for node in range(len(c_ijs)):
        tt_i = theta[user_ids[node]]
        bt_j = beta[question_ids[node]]
        c_ij = c_ijs[node]
        c_j = c[question_ids[node]]

        y = c_j + (1 - c_j) * sigmoid(tt_i - bt_j)
        # Save dJ/dtt=(y-t)(y-c)/(1-c) for the i_th student into 2D array[i][]
        theta_per_stud[user_ids[node]].append(
            (y - c_ij) * (y - c_j) / (1 - c_j))

        # Save dJ/dbt=(t-y)(y-c)/(1-c) for the j_th question into 2D array[j][]
        beta_per_ques[question_ids[node]].append(
            (c_ij - y) * (y - c_j) / (1 - c_j))

        # Save dJ/dc=(y-t)/(y-cy) for the j_th question into 2D array[j][]
        c_per_ques[question_ids[node]].append((y - c_ij) * 1.0 / y / (1 - c_j))

    # tt <- tt - lr * sum(dJ/dtt)   for each student i
    for i in range(len(theta)):
        theta[i] -= lr * np.sum(theta_per_stud[i])

    # bt <- bt - lr * sum(dJ/dbt)   for each question j
    for j in range(len(beta)):
        beta[j] -= lr * np.sum(beta_per_ques[j])

    # c <- c - lr * sum(dJ/dc)   for each question j
    for j in range(len(c)):
        c[j] -= lr * np.sum(c_per_ques[j])

    return theta, beta, c


def irt_2pl(data, val_data, lr, iterations):
    """ Train IRT(2PL) model.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    N = 542  # number of students
    D = 1774  # number of questions
    theta = np.zeros((N,))
    beta = np.zeros((D,))
    c = np.zeros((D,))

    # train_neg_lld_lst = []
    # val_neg_lld_lst = []
    train_acc_lst = []
    val_acc_lst = []

    for i in range(iterations):
        train_acc = evaluate_2pl(data, theta, beta, c)
        val_acc = evaluate_2pl(val_data, theta, beta, c)

        train_acc_lst.append(train_acc)
        val_acc_lst.append(val_acc)

        print(f'{i}_th iter:\n\tTrain Acc:\t{train_acc}\tVal Acc:{val_acc}')
        theta, beta, c = update_theta_beta_c(data, lr, theta, beta, c)

    return theta, beta, c, train_acc_lst, val_acc_lst


def find_best_hyperparams(train_data, val_data, hyperparameters):
    """ Find hyperparameters that give the highest validation accuracy.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param hyperparameters: A dictionary {learning_rates: list,
    num_iterations: list}
    :return: A dictionary {valid_accuracy: float, learning_rate: float,
    num_iteration: float
    """
    max_acc_hyparam = {"valid_accuracy": 0}
    for num_iter in hyperparameters["num_iterations"]:
        for l_rate in hyperparameters["learning_rates"]:
            print(f'\n--------------num_iter={num_iter}-------'
                  f'--------l_rate={l_rate}----------------')
            # Gradient descent on theta and beta
            _, _, _, _, val_acc_lst = irt_2pl(train_data, val_data, l_rate,
                                              num_iter)

            if max(val_acc_lst) > max_acc_hyparam["valid_accuracy"]:
                max_acc_hyparam["learning_rate"] = l_rate
                max_acc_hyparam["num_iteration"] = num_iter
                max_acc_hyparam["valid_accuracy"] = max(val_acc_lst)
                print(max_acc_hyparam)
    return max_acc_hyparam


def evaluate_2pl(data, theta, beta, c):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :param c: Vector
    :return: float
    """
    pred = prediction(data, theta, beta, c)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def prediction(data, theta, beta, c):
    """ Evaluate the model given data and return the prediction.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = c[q] + (1 - c[q]) * sigmoid(x)
        pred.append(p_a >= 0.5)
    return pred


def plot_distribution(lst1, lst2):
    """Helper function to plot histogram

    :param lst1: list of values
    :param lst2: list of values
    """
    plt.figure()
    sns.histplot(lst1, kde=True, bins=100)
    sns.histplot(lst2, kde=True, bins=100)
    plt.axvline(np.mean(lst1), color="b", linestyle="dashed")
    plt.axvline(np.mean(lst2), color='orange', linestyle='dashed')
    _, max_ = plt.ylim()
    plt.text(
        np.mean(lst1) + np.mean(lst1) / 10,
        max_ - max_ / 10,
        "Mean: {:.2f}".format(np.mean(lst1)),
    )
    plt.show()


def compare_2_groups(arr_1, arr_2, alpha=0.05):
    """ T-test that compares the mean of two arrays.

    :param arr_1: list of value
    :param arr_2: list of value
    :param alpha: significance level, default=0.05
    """
    stat, p = ttest_ind(arr_1, arr_2)
    print(f'Statistics={stat:.3f}, p={p:.3f}')
    if p > alpha:
        print('Same distributions (fail to reject H_0)')
    else:
        print('Different distributions (reject H_0)')


def hypo_test(train_data, val_data, test_data):
    """ Hypothesis test comparing mean of two list of accuracy.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param test_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    """
    lr = 0.01
    acc_lst_1pl = []
    acc_lst_2pl = []
    for num_iter in range(50, 70):
        theta_1pl, beta_1pl, _, _, _, _ = irt(train_data, val_data, lr,
                                              num_iter)
        acc_1pl = evaluate_1pl(test_data, theta_1pl, beta_1pl)
        acc_lst_1pl.append(acc_1pl)

        theta_2pl, beta_2pl, c_2pl, _, _ = irt_2pl(train_data, val_data, lr,
                                                   num_iter)
        acc_2pl = evaluate_2pl(test_data, theta_2pl, beta_2pl, c_2pl)
        acc_lst_2pl.append(acc_2pl)

    plot_distribution(acc_lst_1pl, acc_lst_2pl)
    compare_2_groups(acc_lst_1pl, acc_lst_2pl)


def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    hyperparameters = {
        "learning_rates": [0.005, 0.01],
        "num_iterations": [15, 25, 50]
    }
    # max_acc_param = find_best_hyperparams(train_data, val_data, hyperparameters)
    learning_rate = 0.01  # max_acc_param["learning_rate"]
    num_iteration = 50  # max_acc_param["num_iteration"]
    theta, beta, c, train_acc_lst, val_acc_lst \
        = irt_2pl(train_data, val_data, learning_rate, num_iteration)
    train_acc = evaluate(train_data, theta, beta, c)
    val_acc = evaluate(val_data, theta, beta, c)
    test_acc = evaluate(test_data, theta, beta, c)
    print(f'Hyperparameters:\n\tlearning_rate={learning_rate}\tnum_iteration='
          f'{num_iteration}\nFinal Accuracy\n\tTrain:\t{train_acc}\n\tValidation'
          f':\t{val_acc}\n\tTest:\t{test_acc}')

    plt.plot(train_acc_lst, label="Training")
    plt.plot(val_acc_lst, label="Validation")
    plt.xlabel('Iteration Times')
    plt.ylabel('Accuracy')
    plt.title(f'Changes of Accuracy')
    plt.legend()
    plt.show()

    hypo_test(train_data, val_data, test_data)


if __name__ == "__main__":
    main()
