from sklearn.impute import KNNImputer
from utils import *
import numpy as np
import matplotlib.pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html
     for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = matrix.transpose()
    mat = nbrs.fit_transform(mat)
    mat = mat.transpose()
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    ks = [1, 6, 11, 16, 21, 26]
    acc_user = []
    acc_item = []
    for k in ks:
        acc_user.append(knn_impute_by_user(sparse_matrix, val_data, k))
    print("")
    for k in ks:
        acc_item.append(knn_impute_by_item(sparse_matrix, val_data, k))

    plt.xticks(ks)
    plt.plot(ks, acc_item, '.-b', label='item')
    plt.xlabel("k")
    plt.ylabel("val accuracy")
    plt.title("item based knn model")
    plt.savefig('val_acc_item.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.xticks(ks)
    plt.plot(ks, acc_user, '.-r', label='user')
    plt.xlabel("k")
    plt.ylabel("val accuracy")
    plt.title("user based knn model")
    plt.savefig('val_acc_user.png', dpi=300, bbox_inches='tight')
    plt.show()

    user_k = 11
    item_k = 21
    user_test_acc = knn_impute_by_user(sparse_matrix, test_data, user_k)
    item_test_acc = knn_impute_by_item(sparse_matrix, test_data, item_k)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
