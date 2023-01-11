from sklearn.metrics import accuracy_score
from item_response import *
from random import choices


def question_correct_rate(q_id, data):
    df = data.loc[data["question_id"] == q_id]
    n = len(df)
    correct = sum(df["is_correct"])
    rate = correct / n
    print(q_id, ": ", rate, "; subject id: ", df["subject_id"].iloc[0])
    return rate


def user_correct_rate(u_id, data):
    df = data.loc[data["user_id"] == u_id]
    n = len(df)
    correct = sum(df["is_correct"])
    rate = correct / n
    print(u_id, ": ", rate)
    return rate


def bootstrap(data, k, n=3):
    """
    Construct k sub sample of size n from data.
    :param n: size of sub sample
    :param k: number of sub sample, k is smaller than size of data
    :param data: train data.
    :return: k sub sample of size n
    """
    size = len(data["user_id"])
    subsample = [{"user_id": [], "question_id": [], "is_correct": []} for i in range(n)]
    for i in range(n):
        index = choices(range(size), k=k)
        for num in index:
            subsample[i]["user_id"].append(data["user_id"][num])
            subsample[i]["question_id"].append(data["question_id"][num])
            subsample[i]["is_correct"].append(data["is_correct"][num])
    return subsample


def majority_vote(pred1, pred2, pred3):
    """
    :param pred1: pred_y of first model
    :param pred2: pred_y of second model
    :param pred3: pred_y of third model
    :return: pred_y of ensemble model
    """
    pred = []
    for i in range(len(pred1)):
        t = bool(pred1[i]) + bool(pred2[i]) + bool(pred3[i])
        if t >= 2:
            pred.append(1)
        else:
            pred.append(0)
    return pred


def act_main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    y_train = train_data["is_correct"]
    y_val = val_data["is_correct"]
    y_test = test_data["is_correct"]
    train1, train2, train3 = bootstrap(train_data, 30000, 3)

    learning_rate = 0.01
    num_iteration = 15

    theta1, beta1, train_neg_lld1, val_neg_lld1, train_acc_lst1, val_acc_lst1 \
        = irt(train1, val_data, learning_rate, num_iteration)
    theta2, beta2, train_neg_lld2, val_neg_lld2, train_acc_lst2, val_acc_lst2 \
        = irt(train2, val_data, learning_rate, num_iteration)
    theta3, beta3, train_neg_lld3, val_neg_lld3, train_acc_lst3, val_acc_lst3 \
        = irt(train3, val_data, learning_rate, num_iteration)

    train_pred1 = prediction(train_data, theta1, beta1)
    train_pred2 = prediction(train_data, theta2, beta2)
    train_pred3 = prediction(train_data, theta3, beta3)

    val_pred1 = prediction(val_data, theta1, beta1)
    val_pred2 = prediction(val_data, theta2, beta2)
    val_pred3 = prediction(val_data, theta3, beta3)

    test_pred1 = prediction(test_data, theta1, beta1)
    test_pred2 = prediction(test_data, theta2, beta2)
    test_pred3 = prediction(test_data, theta3, beta3)

    train_pred = majority_vote(train_pred1, train_pred2, train_pred3)
    val_pred = majority_vote(val_pred1, val_pred2, val_pred3)
    test_pred = majority_vote(test_pred1, test_pred2, test_pred3)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)
    print(train_acc, val_acc, test_acc)


if __name__ == "__main__":
    act_main()
