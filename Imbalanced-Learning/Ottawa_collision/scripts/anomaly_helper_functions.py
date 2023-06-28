# Model performance metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, auc, recall_score
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer


def power_transformer(n_training_samples, n_test_samples):
    """Power transformation of the training and test sets
    using yeo-johnson method"""

    # Instantiate the class
    power = PowerTransformer(method='yeo-johnson', standardize=True)
    # Fit transform the training set
    X_train_scaled = power.fit_transform(n_training_samples)

    # Only transform the test sets
    X_test_scaled = power.transform(n_test_samples)

    return X_train_scaled, X_test_scaled


def standardizer(n_training_samples, n_test_samples):
    """Standardize the training, validation, and test sets"""

    scaler = StandardScaler()  # Instantiate the class

    # Fit transform the training set
    X_train_scaled = scaler.fit_transform(n_training_samples)

    # Only transform the test sets
    X_test_scaled = scaler.transform(n_test_samples)

    return X_train_scaled, X_test_scaled


def evalaute_performance(y_true, y_pred):
    """ Model prediction """
    # Compute the accuracy of the model
    accuracy = accuracy_score(y_true, y_pred)
    print('Accuracy:  %f' % (accuracy))
    print('AUROC: %f' % (roc_auc_score(y_true, y_pred)))
    print('AUPRC: %f' % (average_precision_score(y_true, y_pred)))
    print('Confusion matrix:\n', confusion_matrix(y_true, y_pred))
    print('Classification report:\n',
          classification_report(y_true, y_pred))
    print('*****************************************************************************')


def Plot_ROC_Curve_and_PRC_Cross(y_true, y_pred, color=None, label=None):
    """ Plot of ROC and PR Curves"""

    # Compute the fpr and tpr for each classifier
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # Compute the precisions and recalls for the classifier
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)

    # Compute the area under the ROC curve for each classifier
    area_auc = roc_auc_score(y_true, y_pred)

    # Compute the area under the PR curve for the classifier
    area_prc = auc(recalls, precisions)

    # ROC Curve
    plt.subplot(121)
    plt.plot(fpr, tpr, color=color, label=(label) % area_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Alarm (FPR)')
    plt.ylabel('Detection Rate (TPR)')
    plt.title('ROC Curve on the Test Set')
    plt.legend(loc='best')

    # PR Curve
    plt.subplot(122)
    plt.plot(recalls, precisions, color=color, label=(label) % area_prc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve  on the Test Set')
    plt.legend(loc='best')
