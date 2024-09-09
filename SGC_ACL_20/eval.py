from haversine import haversine
import numpy as np


def eval(pred_labels, true_labels, user_true_coor, class_lat, class_lon):
    assert len(pred_labels) == len(true_labels), "Number of predictions and true labels must be the same"

    err_distances = [haversine((class_lat[pred_labels[i]], class_lon[pred_labels[i]]),
                               user_true_coor[i]) for i in range(len(pred_labels))]
    accuracy = np.mean([1 if pred_labels[i] == true_labels[i] else 0 for i in range(len(pred_labels))])

    acc_161 = len([dist for dist in err_distances if dist <= 161]) / len(err_distances)
    mean_ed = np.mean(err_distances)
    median_ed = np.median(err_distances)
    return err_distances, accuracy, acc_161, mean_ed, median_ed
