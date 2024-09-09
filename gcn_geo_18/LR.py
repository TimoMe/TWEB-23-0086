from sklearn.linear_model import LinearRegression
import pickle as pkl
import os
from utils import *
import numpy as np


def load_user_data(dir_name, train_start, valid_start, test_start):

    user_mat = pkl.load(open(os.path.join(dir_name, 'user_tfidf_mat_9k.cmu.pkl'), 'rb'))
    user_labels = load_loc_label('cmu')
    train_text_mat = user_mat[train_start:valid_start]
    valid_text_mat = user_mat[valid_start:test_start]
    test_text_mat = user_mat[test_start:]
    train_label = user_labels[train_start:valid_start]
    valid_label = user_labels[valid_start:test_start]
    test_label = user_labels[test_start:]

    return train_text_mat, train_label, valid_text_mat, valid_label, test_text_mat, test_label


def LR(X_train, y_train, X_valid, X_test):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_test = lr.predict(X_test)
    y_pred_valid = lr.predict(X_valid)
    return y_pred_valid, y_pred_test


if __name__ == '__main__':
    # load data
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_user_data('../../dataset/cmu',
                                                                        0, 5685,
                                                                        5685 + 1895)
    # transfer label list to one hot vector
    Y_train = np.zeros((len(y_train), 129))
    for i in range(len(y_train)):
        Y_train[i, y_train[i]] = 1
    print('LR')
    y_pred_valid, y_pred_test = LR(X_train, Y_train, X_valid, X_test)
    print((y_pred_valid))

    # # save numpy.ndarray result
    np.save('LR_cmu_valid.npy', y_pred_valid)
    np.save('LR_cmu_test.npy', y_pred_test)
