# Machine learning models
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Model performance metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, auc, recall_score
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_recall_curve
# Model selection by cross-validation


def model_selection_cv(model, n_training_samples, n_training_labels, cv_fold, scoring=None):

    # Fit the training set
    model.fit(n_training_samples, n_training_labels)

    # Compute accuracy on 10-fold cross validation
    score = cross_val_score(model, n_training_samples, n_training_labels,
                            cv=cv_fold, scoring=scoring)

    # Make prediction on 10-fold cross validation
    y_val_pred = cross_val_predict(
        model, n_training_samples, n_training_labels, cv=cv_fold)

    # Make probability prediction on 10-fold cross validation
    y_pred_proba = cross_val_predict(model, n_training_samples, n_training_labels,
                                     cv=cv_fold, method='predict_proba')[:, 1]

    # Print results
    print('****************************************************************************')
    print('Cross-validation accuracy (std): %f (%f)' %
          (score.mean(), score.std()))
    print('AUROC: %f' % (roc_auc_score(n_training_labels, y_pred_proba)))
    print('AUPRC: %f' % (average_precision_score(n_training_labels, y_pred_proba)))
    print('Predicted classes:', np.unique(y_val_pred))
    print('Confusion matrix:\n', confusion_matrix(
        n_training_labels, y_val_pred))
    print('Classification report:\n', classification_report(
        n_training_labels, y_val_pred))
    print('****************************************************************************')

# Model prediction on the test set


def Test_Prediction(model, n_training_samples, n_training_labels, n_test_samples, n_test_labels):

    # Fit the training set
    model.fit(n_training_samples, n_training_labels)

    # Make prediction on the test set
    y_predict = model.predict(n_test_samples)

    # Compute the accuracy of the model
    accuracy = accuracy_score(n_test_labels, y_predict)

    # Predict probability
    y_predict_proba = model.predict_proba(n_test_samples)[:, 1]

    print('****************************************************************************')
    print('Test accuracy:  %f' % (accuracy))
    print('AUROC: %f' % (roc_auc_score(n_test_labels, y_predict_proba)))
    print('AUPRC: %f' % (average_precision_score(n_test_labels, y_predict_proba)))
    print('Predicted classes:', np.unique(y_predict))
    print('Confusion matrix:\n', confusion_matrix(n_test_labels, y_predict))
    print('Classification report:\n',
          classification_report(n_test_labels, y_predict))
    print('****************************************************************************')

# ROC and PR Curves for the Cross-Validation Training Set


def Plot_ROC_Curve_and_PRC_Cross_Val(model, n_training_samples, n_training_labels, color=None, label=None):

    model.fit(n_training_samples, n_training_labels)

    y_pred_proba = cross_val_predict(model, n_training_samples, n_training_labels, cv=5,
                                     method="predict_proba")

    # Compute the fpr and tpr for each classifier
    fpr, tpr, thresholds = roc_curve(n_training_labels, y_pred_proba[:, 1])

    # Compute the precisions and recalls for the classifier
    precisions, recalls, thresholds = precision_recall_curve(
        n_training_labels, y_pred_proba[:, 1])

    # Compute the area under the ROC curve for each classifier
    area_auc = roc_auc_score(n_training_labels, y_pred_proba[:, 1])

    # Compute the area under the PR curve for the classifier
    area_prc = auc(recalls, precisions)

    # ROC Curve
    plt.subplot(121)
    plt.plot(fpr, tpr, color=color, label=(label) % area_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False positive rate (FPR)')
    plt.ylabel('True positive rate (TPR)')
    plt.title('ROC Curve for the Cross-Validation Training Set')
    plt.legend(loc='best')

    # PR Curve
    plt.subplot(122)
    plt.plot(recalls, precisions, color=color, label=(label) % area_prc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for the Cross-Validation Training Set')
    plt.legend(loc='best')

# ROC and PR Curves


def Plot_ROC_Curve_and_PRC(model, n_training_samples, n_training_labels, n_test_samples, n_test_labels,
                           color=None, label=None):

    # fit the model
    model.fit(n_training_samples, n_training_labels)

    # Predict probability
    y_pred_proba = model.predict_proba(n_test_samples)[:, 1]

    # Compute the fpr and tpr for each classifier
    fpr, tpr, thresholds = roc_curve(n_test_labels, y_pred_proba)

    # Compute the precisions and recalls for the classifier
    precisions, recalls, thresholds = precision_recall_curve(
        n_test_labels, y_pred_proba)

    # Compute the area under the ROC curve for each classifier
    area_auc = roc_auc_score(n_test_labels, y_pred_proba)

    # Compute the area under the PR curve for the classifier
    area_prc = auc(recalls, precisions)

    # ROC Curve
    plt.subplot(121)
    plt.plot(fpr, tpr, color=color, label=(label) % area_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False positive rate (FPR)')
    plt.ylabel('True positive rate (TPR)')
    plt.title('ROC Curve for the Test Set')
    plt.legend(loc='best')

    # PR Curve
    plt.subplot(122)
    plt.plot(recalls, precisions, color=color, label=(label) % area_prc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for the Test Set')
    plt.legend(loc='best')
