import sklearn.metrics
import imblearn.metrics

from .core import register_metric, MetricType


@register_metric(MetricType.PRED, 'accuracy')
def accuracy(y_true, y_pred):
    return sklearn.metrics.accuracy_score(y_true, y_pred)


@register_metric(MetricType.PRED, 'balanced_accuracy')
def balanced_accuracy(y_true, y_pred):
    return sklearn.metrics.balanced_accuracy_score(y_true, y_pred)


@register_metric(MetricType.PRED, 'g_mean')
def g_mean(y_true, y_pred):
    return imblearn.metrics.geometric_mean_score(y_true, y_pred)


@register_metric(MetricType.PROB, 'ROC_AUC')
def roc_auc(y_true, y_prob):
    return sklearn.metrics.roc_auc_score(y_true, y_prob)


@register_metric(MetricType.PRED, "MCC")
def matthews_corrcoef(y_true, y_predict):
    return sklearn.metrics.matthews_corrcoef(y_true, y_predict)


@register_metric(MetricType.PRED, "precision_phish")
def p_precision(y_test, y_predict):
    return sklearn.metrics.precision_score(y_test, y_predict)


@register_metric(MetricType.PRED, "precision_legit")
def l_precision(y_test, y_predict):
    return sklearn.metrics.precision_score(y_test, y_predict, pos_label=0)


@register_metric(MetricType.PRED, "recall_phish")
def p_recall(y_test, y_predict):
    return sklearn.metrics.recall_score(y_test, y_predict)


@register_metric(MetricType.PRED, "recall_legit")
def l_recall(y_test, y_predict):
    return sklearn.metrics.recall_score(y_test, y_predict, pos_label=0)


@register_metric(MetricType.PRED, "f1_score")
def f1_score(y_test, y_predict):
    return sklearn.metrics.f1_score(y_test, y_predict)


@register_metric(MetricType.CLUSTER, "homogenity")
def homogenity(y_test, y_predict):
    return sklearn.metrics.homogeneity_score(y_test, y_predict)


@register_metric(MetricType.CLUSTER, "completeness")
def completeness(y_test, y_predict):
    return sklearn.metrics.completeness_score(y_test, y_predict)


@register_metric(MetricType.CLUSTER, "v_measure")
def v_measure(y_test, y_predict):
    return sklearn.metrics.v_measure_score(y_test, y_predict)