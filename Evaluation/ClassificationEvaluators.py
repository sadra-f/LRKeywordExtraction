import numpy as np

def evaluate_classification(predicted, actual):
    predicted = np.array(predicted)
    actual = np.array(actual)

    classes = np.unique(np.concatenate((predicted, actual)))
    precision_list = []
    recall_list = []
    f1_list = []
    support_list = []

    total_correct = np.sum(predicted == actual)
    accuracy = total_correct / len(actual)

    for cls in classes:
        tp = np.sum((predicted == cls) & (actual == cls))
        fp = np.sum((predicted == cls) & (actual != cls))
        fn = np.sum((predicted != cls) & (actual == cls))
        support = np.sum(actual == cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        precision_list.append(precision * support)
        recall_list.append(recall * support)
        f1_list.append(f1 * support)
        support_list.append(support)

    total_support = np.sum(support_list)
    weighted_precision = np.sum(precision_list) / total_support
    weighted_recall = np.sum(recall_list) / total_support
    weighted_f1 = np.sum(f1_list) / total_support

    return {
        'accuracy': accuracy,
        'precision': weighted_precision,
        'recall': weighted_recall,
        'f1_score': weighted_f1
    }
