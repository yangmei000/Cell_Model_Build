import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import sklearn
import numpy as np
from sklearn import metrics

data_read = pd.read_excel(r"cell_model_build.xlsx")
# print(data_read.columns)
data_read.drop(["Serial number",'ID'], axis=1, inplace=True)
new_columns = [i+"_"+j for i, j in zip(data_read.columns, data_read.iloc[0])]
data_read.columns = new_columns
data_read.drop(index=0, axis=0, inplace=True)
any_null_list = [i for i, j in zip(
    data_read.columns, data_read.isnull().any()) if j == True]
all_null_list = [i for i, j in zip(
    data_read.columns, data_read.isnull().all()) if j == True]
data_null = set(any_null_list).difference(all_null_list)
for i in data_null:
    data_read[i].fillna(np.mean(data_read[i]), inplace=True)
print(data_read.columns)
data_read['Duration of symptoms_1≤7days;\n2>7 days.'] = data_read['Duration of symptoms_1≤7days;\n2>7 days.'].astype(
    'category')
data_read['Duration of symptoms_1≤7days;\n2>7 days.'] = data_read['Duration of symptoms_1≤7days;\n2>7 days.'].apply(
    lambda x: 0 if x == 1 else x)
data_read['Duration of symptoms_1≤7days;\n2>7 days.'] = data_read['Duration of symptoms_1≤7days;\n2>7 days.'].apply(
    lambda x: 1 if x == 2 else x)
y = data_read.get('Duration of symptoms_1≤7days;\n2>7 days.')
X = data_read.drop(['Duration of symptoms_1≤7days;\n2>7 days.'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=30)#30
X_train, X_test, y_train, y_test = x_train.values, x_test.values, y_train.values, y_test.values
eval_set = [(X_train, y_train), (X_test, y_test)]
parameter = {'n_estimators': 15,
                'gamma': 0,
                'max_depth': 5,
                'min_child_weight': 1,
                'learning_rate': 0.15,
                'reg_lambda': 4,
                'use_label_encoder':False,
                'eval_metric':'logloss'}
Model = sklearn.XGBClassifier(**parameter)
eval_set = [(X_train, y_train), (X_test, y_test)]
Model.fit(X_train, y_train, eval_set=eval_set, verbose=True)
y_predict = Model.predict(X_test)
y_prob=Model.predict_proba(X_test)[:,1]
acc = metrics.accuracy_score(y_test, y_predict)
precision = metrics.precision_score(y_test, y_predict)
recall = metrics.recall_score(y_test, y_predict)
f1 = metrics.f1_score(y_test, y_predict)
matrix = metrics.confusion_matrix(y_test, y_predict)
print("Accuracy:"+str(acc))
print("Precision:"+str(precision))
print("Recall:"+str(recall))
print("F1-sorce:"+str(f1))