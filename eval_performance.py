from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from collections import Counter
import numpy as np
import time


def BAE(labels, t_predictions):
    predictions = np.zeros_like(t_predictions)
    predictions[np.arange(np.shape(labels)[0]), np.argmax(t_predictions, axis=1)] = 1
    abs_error = (1 - predictions) * labels  # consider error only for true classes
    freq = np.sum(labels, axis=0) + 1e-15  # count the frequency of each label
    #print(np.shape(labels))
    num_labels = np.shape(labels)[1]
    bae = np.sum(np.sum(abs_error, axis=0) / freq) / num_labels
    # print(bae, np.sum(abs_error, axis=0), freq, num_labels)
    return bae


def evaluate(predictions, labels, multi_label=False):
    #predictions are logits here and binarized labels
    predictions, labels = np.array(predictions), np.array(labels)
    multi_label = False
    if np.sum(labels) > np.shape(labels)[0]:
        multi_label = True

    n_ids, n_labels = np.shape(labels)
    assert predictions.shape == labels.shape, "Shapes: %s, %s" % (predictions.shape, labels.shape,)
    metrics = dict()

    # metrics['cross_entropy'] = -np.mean(labels * np.log(predictions + 1e-15))

    if not multi_label:
        metrics['bae'] = BAE(labels, predictions)
        labels, predictions = np.argmax(labels, axis=1), np.argmax(predictions, axis=1)
        metrics['accuracy'] = accuracy_score(labels, predictions)
        metrics['micro_precision'], metrics['micro_recall'], metrics['micro_f1'], _ = \
            precision_recall_fscore_support(labels, predictions, average='micro')

        metrics['macro_precision'], metrics['macro_recall'], metrics['macro_f1'], metrics['coverage'], \
            metrics['average_precision'], metrics['ranking_loss'], metrics['pak'], metrics['hamming_loss'] \
            = 0, 0, 0, 0, 0, 0, 0, 0

        t_predictions = np.zeros([n_ids, n_labels])
        t_predictions[np.arange(n_ids), predictions] = 1
        predictions = t_predictions

    else:
        metrics['accuracy'] = accuracy_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=1))
        for i in range(predictions.shape[0]):
            k = np.sum(labels[i])
            pos = predictions[i].argsort()
            predictions[i].fill(0)
            predictions[i][pos[-int(k):]] = 1

        metrics['bae'] = 0
        metrics['average_precision'] = 0#label_ranking_average_precision_score(labels, predictions)
        metrics['pak'] = 0#patk(predictions, labels)

        metrics['coverage'] = coverage_error(labels, predictions)
        metrics['ranking_loss'] = label_ranking_loss(labels, predictions)
        metrics['hamming_loss'] = hamming_loss(labels, predictions)
        metrics['micro_precision'], metrics['micro_recall'], metrics['micro_f1'], _ = precision_recall_fscore_support(labels, predictions, average='micro')
        metrics['macro_precision'], metrics['macro_recall'], metrics['macro_f1'], _ = precision_recall_fscore_support(labels, predictions, average='macro')

    return metrics

#
# def test():
#     pred= np.random.rand(15000, 100)
#
#     ground = np.random.rand(15000, 100)
#     ground = ground > 0.5
#     ground = ground.astype(int)
#
#     t0 = time.time()
#     print(evaluate(pred, ground))
#     print(time.time()-t0)

# test()