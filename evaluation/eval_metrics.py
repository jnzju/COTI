import numpy as np
import torch


def precision_recall_f1(pred, target, average_mode='macro', thrs=None):
    """Calculate precision, recall and f1 score according to the prediction and
         target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (float | tuple[float], optional): Predictions with scores under
            the thresholds are considered negative. Default to None.

    Returns:
        float | np.array | list[float | np.array]: Precision, recall, f1 score.
            If the ``average_mode`` is set to macro, np.array is used in favor
            of float to give class-wise results. If the ``average_mode`` is set
             to none, float is used to return a single value.
            If ``thrs`` is a single float or None, the function will return
            float or np.array. If ``thrs`` is a tuple, the function will return
             a list containing metrics for each ``thrs`` condition.
    """

    allowed_average_mode = ['macro', 'none']
    if average_mode not in allowed_average_mode:
        raise ValueError(f'Unsupport type of averaging {average_mode}.')

    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    if isinstance(target, torch.Tensor):
        target = target.numpy()
    assert (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)),\
        (f'pred and target should be torch.Tensor or np.ndarray, '
         f'but got {type(pred)} and {type(target)}.')

    if thrs is None:
        thrs = 0.0
    if isinstance(thrs, float):
        thrs = (thrs, )
        return_single = True
    elif isinstance(thrs, tuple):
        return_single = False
    else:
        raise TypeError(
            f'thrs should be float or tuple, but got {type(thrs)}.')

    label = np.indices(pred.shape)[1]
    pred_label = np.argsort(pred, axis=1)[:, -1]
    pred_score = np.sort(pred, axis=1)[:, -1]

    precisions = []
    recalls = []
    f1_scores = []
    for thr in thrs:
        # Only prediction values larger than thr are counted as positive
        _pred_label = pred_label.copy()
        if thr is not None:
            _pred_label[pred_score <= thr] = -1
        pred_positive = label == _pred_label.reshape(-1, 1)
        gt_positive = label == target.reshape(-1, 1)
        precision = (pred_positive & gt_positive).sum(0) / np.maximum(
            pred_positive.sum(0), 1) * 100
        recall = (pred_positive & gt_positive).sum(0) / np.maximum(
            gt_positive.sum(0), 1) * 100
        f1_score = 2 * precision * recall / np.maximum(precision + recall,
                                                       1e-20)
        if average_mode == 'macro':
            precision = float(precision.mean())
            recall = float(recall.mean())
            f1_score = float(f1_score.mean())
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    if return_single:
        return precisions[0], recalls[0], f1_scores[0]
    else:
        return precisions, recalls, f1_scores


def precision(pred, target, average_mode='macro', thrs=None):
    """Calculate precision according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (float | tuple[float], optional): Predictions with scores under
            the thresholds are considered negative. Default to None.

    Returns:
         float | np.array | list[float | np.array]: Precision.
            If the ``average_mode`` is set to macro, np.array is used in favor
            of float to give class-wise results. If the ``average_mode`` is set
             to none, float is used to return a single value.
            If ``thrs`` is a single float or None, the function will return
            float or np.array. If ``thrs`` is a tuple, the function will return
             a list containing metrics for each ``thrs`` condition.
    """
    precisions, _, _ = precision_recall_f1(pred, target, average_mode, thrs)
    return precisions


def recall(pred, target, average_mode='macro', thrs=None):
    """Calculate recall according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (float | tuple[float], optional): Predictions with scores under
            the thresholds are considered negative. Default to None.

    Returns:
         float | np.array | list[float | np.array]: Recall.
            If the ``average_mode`` is set to macro, np.array is used in favor
            of float to give class-wise results. If the ``average_mode`` is set
             to none, float is used to return a single value.
            If ``thrs`` is a single float or None, the function will return
            float or np.array. If ``thrs`` is a tuple, the function will return
             a list containing metrics for each ``thrs`` condition.
    """
    _, recalls, _ = precision_recall_f1(pred, target, average_mode, thrs)
    return recalls


def f1_score(pred, target, average_mode='macro', thrs=None):
    """Calculate F1 score according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (float | tuple[float], optional): Predictions with scores under
            the thresholds are considered negative. Default to None.

    Returns:
         float | np.array | list[float | np.array]: F1 score.
            If the ``average_mode`` is set to macro, np.array is used in favor
            of float to give class-wise results. If the ``average_mode`` is set
             to none, float is used to return a single value.
            If ``thrs`` is a single float or None, the function will return
            float or np.array. If ``thrs`` is a tuple, the function will return
             a list containing metrics for each ``thrs`` condition.
    """
    _, _, f1_scores = precision_recall_f1(pred, target, average_mode, thrs)
    return f1_scores
