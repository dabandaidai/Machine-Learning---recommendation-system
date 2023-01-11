from utils import *

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    user_ids = data["user_id"]
    question_ids = data["question_id"]
    c_ijs = data["is_correct"]

    # Precondition: len(user_ids) == len(question_ids) == len(c_ijs)
    for node in range(len(c_ijs)):
        theta_i = theta[user_ids[node]]
        beta_j = beta[question_ids[node]]
        c_ij = c_ijs[node]
        # l(tt_i, bt_j) = sum[c_ij(tt_i - bt_j) - log(1+exp(tt_i - bt_j)]
        log_lklihood += c_ij * (theta_i - beta_j) - np.log(
            1 + np.exp(theta_i - beta_j))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # Implement the function as described in the docstring.             #
    #####################################################################
    user_ids = data["user_id"]
    question_ids = data["question_id"]
    c_ijs = data["is_correct"]
    theta_per_stud = [[] for i in range(len(theta))]
    beta_per_ques = [[] for i in range(len(beta))]

    # Precondition:
    #   len(user_ids) == len(question_ids) == len(c_ijs)
    #   user_id in [0, N) and question_id in [0, M)
    for node in range(len(c_ijs)):
        tt_i = theta[user_ids[node]]
        bt_j = beta[question_ids[node]]
        c_ij = c_ijs[node]

        # Save dJ/dtt=y-t for the i_th student into 2D array[i][]
        theta_per_stud[user_ids[node]].append(sigmoid(tt_i - bt_j) - c_ij)

        # Save dJ/dbt=t-y for the j_th question into 2D array[j][]
        beta_per_ques[question_ids[node]].append(c_ij - sigmoid(tt_i - bt_j))

    # tt <- tt - lr * sum(dJ/dtt)   for each student i
    for i in range(len(theta)):
        theta[i] -= lr * np.sum(theta_per_stud[i])

    # bt <- bt - lr * sum(dJ/dbt)   for each question j
    for j in range(len(beta)):
        beta[j] -= lr * np.sum(beta_per_ques[j])
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: tuple of vectors
    """
    N = 542  # number of students
    D = 1774  # number of questions
    theta = np.zeros((N,))
    beta = np.zeros((D,))

    train_neg_lld_lst = []
    val_neg_lld_lst = []
    train_acc_lst = []
    val_acc_lst = []
    for i in range(iterations):
        train_neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        train_acc = evaluate_1pl(data=data, theta=theta, beta=beta)
        val_acc = evaluate_1pl(data=val_data, theta=theta, beta=beta)

        train_neg_lld_lst.append(train_neg_lld)
        val_neg_lld_lst.append(val_neg_lld)
        train_acc_lst.append(train_acc)
        val_acc_lst.append(val_acc)
        print(f'{i}_th iteration:\n\tTrain NLLK: {train_neg_lld}\tTrain Acc:\t'
              f'{train_acc} \n\tVal NLLK:\t{val_neg_lld}\tVal Acc:\t{val_acc}')
        theta, beta = update_theta_beta(data, lr, theta, beta)

    return theta, beta, train_neg_lld_lst, val_neg_lld_lst, \
           train_acc_lst, val_acc_lst


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
            _, _, _, _, _, val_acc_lst = irt(train_data, val_data, l_rate,
                                             num_iter)
            # TODO: max(val_acc_lst)? why use list here
            if max(val_acc_lst) > max_acc_hyparam["valid_accuracy"]:
                max_acc_hyparam["learning_rate"] = l_rate
                max_acc_hyparam["num_iteration"] = num_iter
                max_acc_hyparam["valid_accuracy"] = max(val_acc_lst)
                print(max_acc_hyparam)
    return max_acc_hyparam


def evaluate_1pl(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = prediction(data, theta, beta)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def prediction(data, theta, beta):
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
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return pred


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    hyperparameters = {
        "learning_rates": [0.005, 0.01],
        "num_iterations": [15, 25, 50]
    }
    max_acc_param = find_best_hyperparams(train_data, val_data, hyperparameters)
    learning_rate = max_acc_param["learning_rate"]
    num_iteration = max_acc_param["num_iteration"]
    theta, beta, train_neg_lld, val_neg_lld, train_acc_lst, val_acc_lst \
        = irt(train_data, val_data, learning_rate, num_iteration)
    train_acc = evaluate_1pl(train_data, theta, beta)
    val_acc = evaluate_1pl(val_data, theta, beta)
    test_acc = evaluate_1pl(test_data, theta, beta)
    print(f'Hyperparameters:\n\tlearning_rate={learning_rate}\tnum_iteration='
          f'{num_iteration}\nFinal Accuracy\n\tTrain:\t{train_acc}\n\tValidation'
          f':\t{val_acc}\n\tTest:\t{test_acc}')

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_acc_lst, label="Training")
    plt.plot(val_acc_lst, label="Validation")
    plt.xlabel('Iteration Times')
    plt.ylabel('Accuracy')
    plt.title(f'Changes of Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_neg_lld, label="Training")
    plt.plot(val_neg_lld, label="Validation")
    plt.xlabel('Iteration Times')
    plt.ylabel('Negative Log-Likelihood')
    plt.title(f'Changes of Negative Log-Likelihood')
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # Implement part (d)                                                #
    #####################################################################
    sample_questions = sorted(np.random.choice(1774, 3, replace=True))
    theta = np.linspace(min(theta) - 5, max(theta) + 5, 1000)
    for ques in sample_questions:
        beta_j = beta[ques]
        y = sigmoid(theta - beta_j)
        plt.plot(theta, y, label=f'Question {str(ques)} with beta={beta_j:.6f}')
    plt.legend()
    plt.title("Probability of Correct Response Given Theta")
    plt.xlabel("Theta")
    plt.ylabel("Probability of Correct Response")
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
