import os
import random
import pandas as pd
import gzip
import sys
import io
import numpy as np

#--- parameter setting
wkdir = sys.argv[1]

group_method = sys.argv[2] # k_fold: binary class, stratified_kfold: multiple class
num_splits = int(sys.argv[3])
num_repeats = int(sys.argv[4])

feature_method = sys.argv[5]
num_features = int(sys.argv[6])

class_model = sys.argv[7]
class_label = sys.argv[8]

folder_path = sys.argv[9] # for save the results

input_X_file = sys.argv[10]
input_y_file = sys.argv[11]

input_X_test_file = sys.argv[12]
#-------------------------------------
#--- define the work path
os.chdir(wkdir)
#--- define the path
func_path = "{0}/functions".format(wkdir)
sys.path.append(func_path)
import classification_function as cf

input_path = "{0}/data".format(wkdir)

# folder_path = "Pig.sex.counts.Saturated.results"
os.makedirs(folder_path, exist_ok = True)
output_path = "{0}/{1}".format(wkdir, folder_path)
output_file = "{0}.F{1}.{2}.L_{3}.csv".format(feature_method, num_features, class_model, class_label)
featur_file = "{0}.F{1}.{2}.L_{3}.feature.csv".format(feature_method, num_features, class_model, class_label)
predic_file = "{0}.F{1}.{2}.L_{3}.predict.csv".format(feature_method, num_features, class_model, class_label)

#--- read gene expression data
with gzip.open("{0}/{1}".format(input_path, input_X_file), 'rt') as file:
    X = file.read()

#--- transfer data from str to dataframe
X = pd.read_csv(io.StringIO(X), delimiter=' ', header = 0)
#--- get rowname and colname
ind_id = X.index.tolist()
gene_id = X.columns.tolist()
# log2(expr + 1) transfer 
X = np.log2(X + 1)
#--- transfer from data.frame to array
X = X.values

#--- read label data
with open("{0}/{1}".format(input_path, input_y_file), 'r') as file:
    y = file.read()

y = pd.read_csv(io.StringIO(y), delimiter=' ', header = 0)
y = y["label"]
y = y.values

#--- create data frame f for cv group
f = np.column_stack((ind_id, y))

#---------------------------------------------------
#--- read X_test, y_test data
#--- when we predict the unknown label, we need to read the predicted data
#---------------------------------------------------
#--- read gene expression data
if class_label == "Unknown":
    with gzip.open("{0}/{1}".format(input_path, input_X_test_file), 'rt') as file:
        X_test = file.read()
    #--- transfer data from str to dataframe
    X_test = pd.read_csv(io.StringIO(X_test), delimiter=' ', header = 0)
    #--- get rowname and colname
    test_ind_id = X_test.index.tolist()
    # gene_id = X.columns.tolist()
    # log2(expr + 1) transfer 
    X_test = np.log2(X_test + 1)
    #--- transfer from data.frame to array
    X_test = X_test.values
    #--- set the y_test
    y_test = np.full(len(X_test), "Unknown")



#--- save the prediction results
df_titles = pd.DataFrame({
    'Repeat': [],
    'Fold': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1': [],
    'feature_select_time': [],
    'predict_time': []
})
df_titles.to_csv("{0}/{1}".format(output_path, output_file), index = False)

#--- save the feature information
f_titiles = pd.DataFrame({
        'Repeat': [],
        'Fold': [],
        'gene_id': []
    })
f_titiles.to_csv("{0}/{1}".format(output_path, featur_file), index = False)

#--- save the prediction results
cv_titiles = pd.DataFrame({
        'Repeat': [],
        'Fold': [],
        'ind_id': [],
        'True_label': [],
        'Pred_label': []
    })
cv_titiles.to_csv("{0}/{1}".format(output_path, predic_file), index = False)

# data grouped
splits = cf.split_data(X, y, f, method = group_method, num_splits = num_splits, num_repeats = num_repeats)

for i, (train_index, test_index) in enumerate(splits):
    repeat = (i // num_splits) + 1
    fold = (i % num_splits) + 1
    print(f"Repeat {repeat}, Fold {fold}:")
    #--- divide dataset
    if class_label != "Unknown":
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    if class_label == "Unknown":
        X_train = X[train_index]
        y_train = y[train_index]
    #--- feature selection
    X_train_selected, X_test_selected, selected_feature_indices, feature_select_time = cf.select_features(X_train, y_train, X_test, method = feature_method, num_features = num_features)
    #--- clasification
    accuracy, cm, report, precision, recall, f1, predict_time, y_pred, model = cf.predict(X_train_selected, y_train, X_test_selected, y_test, class_label = class_label, group_method = group_method, model = class_model)
    #--- save the results
    df_i = pd.DataFrame({
        'Repeat': [repeat],
        'Fold': [fold],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1': [f1],
        'feature_select_time': [feature_select_time],
        'predict_time': [predict_time]
    })
    df_i.to_csv("{0}/{1}".format(output_path, output_file), mode='a', header = False, index = False)
    #--- save feature result
    f_i = {
        'Repeat': np.full(len(selected_feature_indices), repeat), 
        'Fold': np.full(len(selected_feature_indices), fold), 
        'gene_id': np.array([gene_id[i] for i in selected_feature_indices])
    }
    f_i = pd.DataFrame(f_i)
    f_i.to_csv("{0}/{1}".format(output_path, featur_file), mode='a', header = False, index = False)
    #--- save predict result
    if class_label == "Unknown":
        cv_i = {
        'Repeat': np.full(len(test_ind_id), repeat), 
        'Fold': np.full(len(test_ind_id), fold), 
        'ind_id': np.array(test_ind_id), 
        'True_label': y_test,
        'Pred_label': y_pred
        }
    if class_label != "Unknown":
        cv_i = {
        'Repeat': np.full(len(test_index), repeat), 
        'Fold': np.full(len(test_index), fold), 
        'ind_id': np.array([ind_id[i] for i in test_index]), 
        'True_label': y_test,
        'Pred_label': y_pred
        }
    cv_i = pd.DataFrame(cv_i)
    cv_i.to_csv("{0}/{1}".format(output_path, predic_file), mode = 'a', header = False, index = False)

