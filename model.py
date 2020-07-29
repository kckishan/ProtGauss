import pickle as pkl
import os
from os.path import isfile, isdir
import numpy as np
import pandas as pd
from utils.utils import create_matrices
from utils.gaussian import *
from utils.evaluation import *
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
import warnings; warnings.filterwarnings("ignore")
from prettytable import PrettyTable
from tqdm import tqdm

def run_model(org, folder_name, processedDataFile, args, classes, selected_classes):
    """

    Parameters
    ----------
    org : dataset name
    folder_name : folder to save the results
    processedDataFile : Preprocessed data
    args : Argument parser object
    classes : Nucleosome binding modes
    selected_classes : Class selected for training

    Returns
    -------

    """
    with open(processedDataFile, "rb") as filename:
        data = pkl.load(filename)

    train_emb = data['train_emb']
    test_emb  = data['test_emb']
    train_labels =  data['train_labels'].astype(int)
    print("Training embeddings shape:", train_emb.shape)
    print("Testing embeddings shape:", test_emb.shape)
    print("Training labels shape:", train_labels.shape)


    # Remove the TFs if it does not have any binding modes
    del_rid = np.where(train_labels.sum(axis=1) == 0)[0]
    train_emb = np.delete(train_emb, del_rid, axis=0)
    train_labels = np.delete(train_labels, del_rid, axis=0)
    train_labels = np.array(train_labels)

    # display the number of positive and negatives samples on training binding modes
    class_distribution = PrettyTable()
    class_distribution.field_names = ["class", "True", "False"]

    # select the configured binding modes
    classes = list(map(classes.__getitem__, selected_classes))
    train_labels = train_labels[:, selected_classes]
    print(train_labels.shape)
    num_classes = len(classes)

    for k, v in enumerate(classes):
        class_distribution.add_row([v, sum(train_labels[:,k]), len(train_labels[:,k])-sum(train_labels[:,k])])

    print(class_distribution)

    # create the matrix representation X for each TFs
    train_matrices = create_matrices(train_emb)
    test_matrices = create_matrices(test_emb)
    output = {}

    best_alpha = 0
    max_micro_aupr = 0
    max_acc = 0
    c_penalty = 10
    num_splits = 10
    print("Cross Validation with different alpha")
    for alpha in tqdm(range(0, 11, 1)):
        # create the kernal matrix as discussed on the paper
        kernel_matrix_train, kernel_matrix_test = build_kernel_matrices(train_matrices, test_matrices, 100, alpha=alpha/10, run_test=False)

        f_macro_aupr = []
        f_micro_aupr = []
        f_acc = []
        f_f1 = []
        # repeat the experiment 10 times
        for seed in range(10):
            # 5 fold cross validation splits
            # defined seed based on for loop for reproducibility
            splits = ml_split(train_labels, num_splits=num_splits, seed=seed)
            macro_aupr = []
            micro_aupr = []
            acc = []
            f1 = []
            counter = 0
            for train, valid in splits:
                X_train = kernel_matrix_train[train, :][:, train]
                X_valid = kernel_matrix_train[valid, :][:, train]

                y_train = train_labels[train.astype(int)]
                y_val = train_labels[valid.astype(int)]

                # choose whether to train multi-label or binary classifier
                if num_classes == 1:
                    clf = svm.SVC(C=c_penalty, kernel='precomputed', random_state=42, probability=True)
                else:
                    clf = OneVsRestClassifier(svm.SVC(C=c_penalty, kernel='precomputed', random_state=42, probability=True), n_jobs=-1)
                clf = clf.fit(X_train, y_train)

                # for multilabel classification
                if num_classes > 1:
                    y_score_valid = clf.predict_proba(X_valid)
                    y_pred_valid =  clf.predict(X_valid)
                    result = evaluate_performance(y_val, np.array(y_score_valid), np.array(y_pred_valid), 3)
                else:
                    # for binary classification
                    y_score_valid = clf.predict_proba(X_valid)[:,1].reshape(y_val.shape[0], -1)
                    y_pred_valid =  clf.predict(X_valid).reshape(y_val.shape[0], -1)
                    result = evaluate_performance(y_val, np.array(y_score_valid), np.array(y_pred_valid), 1)

                micro_aupr.append(result['m-aupr'])
                macro_aupr.append(result['M-aupr'])
                f1.append(result['F1'])
                acc.append(result['acc'])
                counter +=1

            f_micro_aupr.append(round(np.mean(micro_aupr), 3))
            f_macro_aupr.append(round(np.mean(macro_aupr), 3))
            f_f1.append(round(np.mean(f1), 3))
            f_acc.append(round(np.mean(acc), 3))

        output[alpha] = {}
        output[alpha]['mean_micro'] = round(np.mean(f_micro_aupr), 3)
        output[alpha]['mean_macro'] =  round(np.mean(f_macro_aupr), 3)
        output[alpha]['mean_f1'] =  round(np.mean(f_f1), 3)
        output[alpha]['mean_acc'] =  round(np.mean(f_acc), 3)
        output[alpha]['std_micro'] = round(np.std(f_micro_aupr), 3)
        output[alpha]['std_macro'] =  round(np.std(f_macro_aupr), 3)
        output[alpha]['std_f1'] =  round(np.std(f_f1), 3)
        output[alpha]['std_acc'] =  round(np.std(f_acc), 3)

        # choose alpha based on micro-AUPR
        if max_micro_aupr <= round(np.mean(f_micro_aupr), 3) and max_acc <= round(np.mean(f_acc), 3):
            max_micro_aupr = round(np.mean(f_micro_aupr), 3)
            max_acc = round(np.mean(f_acc), 3)
            best_alpha = alpha/10

    print("The optimum value of alpha:", best_alpha)

    df = pd.DataFrame(output)
    df.columns = np.arange(0, 1.1, 0.1)
    df = df.transpose()
    df = df.reset_index()
    df.columns = ["alpha", "mean_micro", "mean_macro", "mean_f1", "mean_acc", "std_micro", "std_macro", "std_f1", "std_acc"]

    # display the performance for different alphas
    print(df)
    out_folder = "results/"+str(args.seqtype)+"/"
    if not isdir(out_folder):
        os.makedirs(out_folder)

    # save the performance results
    if num_classes == 1:
        file_name = out_folder+"_".join(classes)+"_"+str(args.len)+".csv"
    elif num_classes == 7:
        file_name = out_folder+"all_"+str(args.len)+".csv"
    df.to_csv(file_name, index=False)

    # using the optimum value of alpha, retrain the model with all training sequences
    # and evaluate on test sequences
    kernel_matrix_train, kernel_matrix_test = build_kernel_matrices(train_matrices, test_matrices, 100, alpha=best_alpha, run_test=True)

    clf = svm.SVC(C=c_penalty, kernel='precomputed', random_state=42, probability=True)
    if num_classes > 1:
        clf = OneVsRestClassifier(clf, n_jobs=-1)

    clf = clf.fit(kernel_matrix_train, train_labels)
    y_pred = clf.predict(kernel_matrix_test)

    #saving the prediction
    if num_classes == 1:
        dec = clf.decision_function(kernel_matrix_test)
        prob = 1./(1 + np.exp(-dec))
        assert np.all((prob > 0.5) == y_pred)
        prediction_df = pd.DataFrame()
        prediction_df['tf'] = data['test_tf']
        prediction_df['probability'] = prob.reshape(-1)
        prediction_df['pred'] = y_pred.reshape(-1)
        file_name = folder_name + org + "_prediction_"+ str(args.seqtype) + str(args.len) + ".csv"
    else:
        tf_df = pd.DataFrame()
        tf_df['tf'] = data['test_tf']
        prob_df = pd.DataFrame(y_pred)
        prediction_df = pd.concat([tf_df, prob_df], axis=1)
        file_name = folder_name+ org + "_prediction_"+str(args.seqtype)+"_all_"+str(args.len)+".csv"

    # save the prediction
    prediction_df.to_csv(file_name, index=False)