# Machine learning models
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.base import TransformerMixin

# Model performance metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, f1_score, auc, recall_score
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_recall_curve


class DataFrameImputer(TransformerMixin):
    """
    This is based on some nice code by 'sveitser' at http://stackoverflow.com/a/25562948

    Impute missing categorical and numerical  values (if any).
    Columns of dtype object are imputed with the most frequent value in column.
    Columns of other types (if any) are imputed with median of column

    """

    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0] if X[c].dtype == np.dtype('O')
                               else X[c].median() for c in X.columns], index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


def model_selection_cv(model, n_training_samples, n_training_labels, cv_fold, scoring=None):
    """ Model selection by cross-validation of binary-class"""
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


def model_evaluation(model, n_training_samples, n_training_labels, cv_fold, scoring=None):
    """ Model selection by cross-validation of multi-class """
    # Fit the training set
    model.fit(n_training_samples, n_training_labels)

    # Compute accuracy on 10-fold cross validation
    score = cross_val_score(model, n_training_samples, n_training_labels,
                            cv=cv_fold, scoring=scoring)

    # Make prediction on 10-fold cross validation
    y_val_pred = cross_val_predict(
        model, n_training_samples, n_training_labels, cv=cv_fold)

    # Compute the accuracy of the model
    f1 = f1_score(n_training_labels, y_val_pred, average='weighted')

    # Confusion matrix
    conf_mx = confusion_matrix(n_training_labels, y_val_pred)

    # Classification report
    class_report = classification_report(n_training_labels, y_val_pred)

    print('****************************************************************************')
    print('Cross-validation accuracy (std): %f (%f)' %
          (score.mean(), score.std()))
    print('f1_score:',  f1)
    print('Confusion matrix:\n', conf_mx)
    print('Classification report:\n', class_report)
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
