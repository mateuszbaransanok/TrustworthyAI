import numpy as np
from sklearn import metrics


def calc_metrics(
    predictions: np.ndarray,
    confidences: np.ndarray,
    labels: np.ndarray,
    ground_truth: np.ndarray,
) -> dict[str, float]:
    auroc, aupr_in, aupr_out = auc(
        confidences=confidences,
        ground_truth=ground_truth,
    )

    ccrs = {
        f'ccr_{i}': ccr(
            predictions=predictions,
            confidences=confidences,
            labels=labels,
            recall=recall,
        )
        for i, recall in enumerate((0.1, 0.01, 0.001, 0.0001), start=1)
    }

    return {
        'accuracy': accuracy(
            predictions=predictions,
            labels=labels,
        ),
        'fpr@95': fpr(
            confidences=confidences,
            ground_truth=ground_truth,
        ),
        'detection_error': detection_error(
            confidences=confidences,
            ground_truth=ground_truth,
        ),
        'auroc': auroc,
        'aupr_in': aupr_in,
        'aupr_out': aupr_out,
        'ind_support': float(ground_truth.sum()),
        'ood_support': float((1 - ground_truth).sum()),
        **ccrs,
    }


def fpr(
    confidences: np.ndarray,
    ground_truth: np.ndarray,
    recall: float = 0.95,
    pos_label: int = 1,
) -> float:
    fpr_list, tpr_list, _ = metrics.roc_curve(ground_truth, confidences, pos_label=pos_label)
    return fpr_list[np.argmax(tpr_list >= recall)]


def auc(
    confidences: np.ndarray,
    ground_truth: np.ndarray,
    pos_label: int = 1,
) -> tuple[float, float, float]:
    fpr_list, tpr_list, _ = metrics.roc_curve(ground_truth, confidences, pos_label=pos_label)

    precision_in, recall_in, _ = metrics.precision_recall_curve(
        y_true=ground_truth,
        probas_pred=confidences,
        pos_label=pos_label,
    )

    precision_out, recall_out, _ = metrics.precision_recall_curve(
        y_true=1 - ground_truth,
        probas_pred=1 - confidences,
        pos_label=pos_label,
    )

    auroc = metrics.auc(fpr_list, tpr_list)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out


def accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> float:
    ind_predictions = predictions[labels != -1]
    ind_labels = labels[labels != -1]

    num_true_positives = np.sum(ind_predictions == ind_labels)
    return num_true_positives / len(ind_labels)


def ccr(
    predictions: np.ndarray,
    confidences: np.ndarray,
    labels: np.ndarray,
    recall: float = 0.95,
) -> float:
    ind_confidences = confidences[labels != -1]
    ind_predictions = predictions[labels != -1]
    ind_labels = labels[labels != -1]

    ood_confidences = confidences[labels == -1]

    num_false_positives = int(np.ceil(recall * len(ood_confidences)))
    threshold = np.sort(ood_confidences)[-num_false_positives]
    num_true_positives = np.sum((ind_confidences > threshold) * (ind_predictions == ind_labels))
    return num_true_positives / len(ind_confidences)


def detection_error(
    confidences: np.ndarray,
    ground_truth: np.ndarray,
    recall: float = 0.95,
    pos_label: int = 1,
) -> float:
    fpr_list, tpr_list, _ = metrics.roc_curve(ground_truth, confidences, pos_label=pos_label)

    pos_ratio = sum(ground_truth == pos_label) / len(ground_truth)
    neg_ratio = 1 - pos_ratio

    def _detection_error(index: int) -> float:
        return neg_ratio * (1 - tpr_list[index]) + pos_ratio * fpr_list[index]

    return min((_detection_error(i) for i, x in enumerate(tpr_list) if x >= recall), default=np.inf)
