from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from SVM_RF_LG.load_data import load_eod_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score,matthews_corrcoef
import warnings
from statistics import mean
import pandas as pd
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore',category=DeprecationWarning)


dataset_name = 'ACL18'
# dataset_name = 'KDD17'

if dataset_name == 'ACL18':
    data_path = '../data/ACL18/ourpped'
elif dataset_name == 'KDD17':
    data_path = '../data/KDD17/ourpped'

# Model SVM/RF/LG
model_name = 'RF'

train_data,train_label,test_data,test_label,tickers = load_eod_data(dataset_name, data_path) # err for kdd17 split
# print(train_data.shape)
acc,f1,pre,rec,auc,mcc = [],[],[],[],[],[]
pred_labels = pd.DataFrame()
true_labels = pd.DataFrame()
for index in range(train_data.shape[0]):
    # generate train/test data for single eod
    tr_data = train_data[index,:,:]
    tr_label = train_label[index,:]
    te_data = test_data[index,:,:]
    te_label = test_label[index,:]
    # creat SVM model for single eod
    if model_name == 'SVM':
        clf = SVC()
        clf.fit(tr_data,tr_label)
        SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
            max_iter=-1, probability=False, random_state=None, shrinking=True,
            tol=0.001, verbose=False)
        y_pred = clf.predict(te_data)
    elif model_name == 'RF':
        clf = RandomForestClassifier(max_depth=2,random_state=0)
        clf.fit(tr_data,tr_label)
        y_pred = clf.predict(te_data)
    elif model_name == 'LG':
        clf = LogisticRegression(penalty='l2',C=1,solver='lbfgs',max_iter=1000)
        clf.fit(tr_data,tr_label)
        y_pred = clf.predict(te_data)
    # save predicted results y_pred to csv file based on stock index
    pred_labels[tickers[index]] = y_pred.tolist()
    true_labels[tickers[index]] = te_label.tolist()
    # # Debug Button
    # if tickers[index] == 'AAPL':
    #     print(te_label.shape)
    #     print(te_label)
    #     print(y_pred.shape)
    #     print(y_pred.tolist())
    ep_accuracy_score = accuracy_score(te_label, y_pred)
    # ep_f1_score = f1_score(te_label, y_pred, average='binary')
    ep_f1_score = f1_score(te_label, y_pred,average='weighted')
    ep_precision_score = precision_score(te_label, y_pred)
    ep_recall_score = recall_score(te_label, y_pred)
    ep_auc_score = roc_auc_score(te_label, y_pred)
    ep_mcc_score = matthews_corrcoef(te_label, y_pred)
    # # Debug Button
    # if tickers[index] == 'AAPL':
    #     print('index:{} stock:{} acc:{} f1:{} pre:{} rec:{} auc:{} mcc:{}'.format(
    #         index, tickers[index], ep_accuracy_score, ep_f1_score, ep_precision_score, ep_recall_score, ep_auc_score, ep_mcc_score
    # ))
    print('index:{} stock:{} acc:{} f1:{} pre:{} rec:{} auc:{} mcc:{}'.format(
        index, tickers[index], ep_accuracy_score, ep_f1_score, ep_precision_score, ep_recall_score, ep_auc_score,
        ep_mcc_score
    ))
    acc.append(ep_accuracy_score)
    f1.append(ep_f1_score)
    pre.append(ep_precision_score)
    rec.append(ep_recall_score)
    auc.append(ep_auc_score)
    mcc.append(ep_mcc_score)
# print(pred_labels)
pred_labels.to_csv('kdd17-rf-predicted.csv',index=None)
# true_labels.to_csv('kdd17-true.csv',index=None)
ACCS = mean(acc)
F1S = mean(f1)
PreS = mean(pre)
RecS = mean(rec)
AUCS = mean(auc)
MCCS = mean(mcc)
print('*************************************************')
print('Acc:{} F1:{} Pre:{} Rec:{} Auc:{} Mcc:{}'.format(
        ACCS, F1S, PreS, RecS, AUCS, MCCS
    ))
print('*************************************************')