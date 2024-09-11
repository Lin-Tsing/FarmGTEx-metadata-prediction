import time
from sklearn.model_selection import RepeatedKFold
# from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, f_classif, mutual_info_classif, f_regression, mutual_info_regression
from sklearn.feature_selection import RFE, RFECV, SelectFromModel, SelectFdr

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, auc

#--- split data for CV
def split_data(X, y, f, method = 'k_fold', num_splits = 5, num_repeats = 20, random_state = 1000):
    if method == 'k_fold':
        kf = RepeatedKFold(n_splits = num_splits, n_repeats = num_repeats, random_state = random_state)
        splits = kf.split(X)
    elif method == 'stratified_kfold':
        # kf = StratifiedKFold(n_splits = num_splits, shuffle = True, random_state = random_state)
        kf = RepeatedStratifiedKFold(n_splits = num_splits, n_repeats = num_repeats, random_state = random_state)
        splits = kf.split(f, f[:, -1])
    else:
        raise ValueError("Invalid data grouping method")

    # data_splits = []
    # for train_index, test_index in splits:
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #     data_splits.append((X_train, X_test, y_train, y_test, test_index))

    return splits

#--- feature selection for training data
def select_features(X_train, y_train, X_test, method='f_classif', num_features = 10):
    start_time = time.time()
    if method == 'f_classif':
        selector = SelectKBest(f_classif, k = num_features)
    elif method == 'chi2':
        selector = SelectKBest(chi2, k=num_features)
    elif method == 'select_from_model':
        estimator = LogisticRegression(n_jobs=-1)
        selector = SelectFromModel(estimator, max_features=num_features)
    elif method == 'rfe':                           # it is too slow!
        estimator = LogisticRegression(n_jobs=-1)
        selector = RFE(estimator, n_features_to_select=num_features)
    elif method == 'percentile':                    # the parameter is different with others
        selector = SelectPercentile(f_classif, percentile=num_features)
    elif method == 'mutual_info_classif':           # it is too slow!
        selector = SelectKBest(mutual_info_classif, k=num_features)
    elif method == 'f_regression':                  # not classification
        selector = SelectKBest(f_regression, k=num_features)
    elif method == 'mutual_info_regression':        # not classification
        selector = SelectKBest(mutual_info_regression, k=num_features)
    elif method == 'rfecv':
        estimator = LogisticRegression(n_jobs=-1)   # it is not fit classification
        selector = RFECV(estimator, step=1, cv=5)
    else:
        raise ValueError("Invalid feature selection method")

    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    selected_feature_indices = selector.get_support(indices = True)
    
    end_time = time.time()
    feature_select_time = end_time - start_time

    return X_train_selected, X_test_selected, selected_feature_indices, feature_select_time


def predict(X_train_selected, y_train, X_test_selected, y_test, class_label, group_method = "k_fold", model='logistic_regression'):
    if model == 'logistic_regression':
        model = make_pipeline(StandardScaler(), LogisticRegression(n_jobs=-1))
    elif model == 'svc':
        model = make_pipeline(StandardScaler(), SVC())
    elif model == 'decision_tree':
        model = make_pipeline(StandardScaler(), DecisionTreeClassifier())
    elif model == 'k_neighbors':
        model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_jobs=-1))
    elif model == 'gaussian_nb':
        model = make_pipeline(StandardScaler(), GaussianNB())
    elif model == 'mlp':
        model = make_pipeline(StandardScaler(), MLPClassifier())
    elif model == 'random_forest':
        model = make_pipeline(StandardScaler(), RandomForestClassifier(n_jobs=-1))
    elif model == 'gradient_boosting':
        model = make_pipeline(StandardScaler(), GradientBoostingClassifier())
    elif model == 'adaboost':
        model = make_pipeline(StandardScaler(), AdaBoostClassifier())
    elif model == 'bagging':
        model = make_pipeline(StandardScaler(), BaggingClassifier(n_jobs=-1))
    elif model == 'extra_trees':
        model = make_pipeline(StandardScaler(), ExtraTreesClassifier(n_jobs=-1))
    elif model == 'voting':
        model = make_pipeline(StandardScaler(), VotingClassifier(
            estimators=[('lr', LogisticRegression(n_jobs=-1)), ('rf', RandomForestClassifier(n_jobs=-1)), ('gb', GradientBoostingClassifier())]
        ))
    else:
        raise ValueError("Invalid model")

    start_time = time.time()
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    end_time = time.time()

    predict_time = end_time - start_time

    if(class_label == "Unknown"):
        accuracy = None
        cm = None
        report = None
        precision = None
        recall = None
        f1 = None
    else:
        if group_method == 'k_fold':
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            precision = precision_score(y_test, y_pred, pos_label = class_label)
            recall = recall_score(y_test, y_pred, pos_label = class_label)
            f1 = f1_score(y_test, y_pred, pos_label = class_label)
            # fpr, tpr, thresholds = roc_curve(y_test_num, y_pred_num)
        elif group_method == 'stratified_kfold':
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            precision = precision_score(y_test, y_pred, pos_label = class_label, average = 'weighted')
            recall = recall_score(y_test, y_pred, pos_label = class_label, average = 'weighted')
            f1 = f1_score(y_test, y_pred, pos_label = class_label, average = 'weighted')
            # fpr, tpr, thresholds = roc_curve(y_test_num, y_pred_num)
        else:
            raise ValueError("Invalid data grouping method")
    
    # return accuracy, cm, report, precision, recall, f1, predict_time
    return accuracy, cm, report, precision, recall, f1, predict_time, y_pred, model

# #--- predict the Unknown label
# def predict_unknown(X_train_selected, y_train, X_test_selected, model='logistic_regression'):
#     if model == 'logistic_regression':
#         model = make_pipeline(StandardScaler(), LogisticRegression(n_jobs=-1))
#     elif model == 'svc':
#         model = make_pipeline(StandardScaler(), SVC())
#     elif model == 'decision_tree':
#         model = make_pipeline(StandardScaler(), DecisionTreeClassifier())
#     elif model == 'k_neighbors':
#         model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_jobs=-1))
#     elif model == 'gaussian_nb':
#         model = make_pipeline(StandardScaler(), GaussianNB())
#     elif model == 'mlp':
#         model = make_pipeline(StandardScaler(), MLPClassifier())
#     elif model == 'random_forest':
#         model = make_pipeline(StandardScaler(), RandomForestClassifier(n_jobs=-1))
#     elif model == 'gradient_boosting':
#         model = make_pipeline(StandardScaler(), GradientBoostingClassifier())
#     elif model == 'adaboost':
#         model = make_pipeline(StandardScaler(), AdaBoostClassifier())
#     elif model == 'bagging':
#         model = make_pipeline(StandardScaler(), BaggingClassifier(n_jobs=-1))
#     elif model == 'extra_trees':
#         model = make_pipeline(StandardScaler(), ExtraTreesClassifier(n_jobs=-1))
#     elif model == 'voting':
#         model = make_pipeline(StandardScaler(), VotingClassifier(
#             estimators=[('lr', LogisticRegression(n_jobs=-1)), ('rf', RandomForestClassifier(n_jobs=-1)), ('gb', GradientBoostingClassifier())]
#         ))
#     else:
#         raise ValueError("Invalid model")

#     start_time = time.time()
#     model.fit(X_train_selected, y_train)
#     y_pred = model.predict(X_test_selected)
#     end_time = time.time()

#     predict_time = end_time - start_time

#     # return accuracy, cm, report, precision, recall, f1, predict_time
#     return model, predict_time, y_pred

