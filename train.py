import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data_processed.csv", index_col=False)

X_sm = df.drop(columns = ['stroke'])
y_sm = df['stroke']

# Split
X_train0, X_test0, y_train, y_test = train_test_split(
    X_sm,
    y_sm,
    test_size = .2,
    random_state = 777)
X_train0.shape, y_train.shape, X_test0.shape, y_test.shape

scaler = StandardScaler()
scaler.fit(X_train0)
X_train = scaler.transform(X_train0)
X_test = scaler.transform(X_test0)

clf_rf = RandomForestClassifier(random_state=777)
clf_rf = clf_rf.fit(X_train,y_train)
y_pred_rf = clf_rf.predict(X_test)
acc = accuracy_score(y_test, y_pred_rf)
print('Testing-set Accuracy score is:', acc)
print('Training-set Accuracy score is:',accuracy_score(y_train,clf_rf.predict(X_train)))
cm_rf = confusion_matrix(y_test, y_pred_rf)
# sns.heatmap(cm_rf, annot = True, fmt = "d")

gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
acc = accuracy_score(y_test, gb_pred)
print("Gradient Boosting Classifier Model Accuracy score is:", acc)
cm_gb = confusion_matrix(y_test, gb_pred)
# sns.heatmap(cm_gb, annot = True, fmt="d")

knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
acc = knn.score(X_test, y_test)
print("KNN Model Acuuracy is:", acc)
cm_knn = confusion_matrix(y_test, knn_pred)
# sns.heatmap(cm_knn, annot = True, fmt="d")

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
acc = lr.score(X_test, y_test)
print("LogisticRegression accuracy score is:",acc)
report = classification_report(y_test, lr_pred)
print(report)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
acc = accuracy_score(y_test, dt_pred)
print("Decision Tree accuracy score is :",acc)
cm_dt = confusion_matrix(y_test, dt_pred)
# sns.heatmap(cm_dt, annot = True, fmt = "d")

clf1 = GradientBoostingClassifier()
clf2 = GradientBoostingClassifier()
clf3 =  RandomForestClassifier()
eclf1 = VotingClassifier(estimators=[('gbc', clf1),('gbc2', clf2), ('rf2', clf3) ], voting='soft')
eclf1.fit(X_train, y_train)
predictions = eclf1.predict(X_test)
print("Voting Classifier Accuracy Score is: ")
print(accuracy_score(y_test, predictions))
cm = confusion_matrix(y_test, predictions)
# sns.heatmap(cm, annot = True, fmt="d")

acc = accuracy_score(y_test, predictions)

tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

ax = sns.heatmap(cm, annot = True, fmt="d")
ax.set_title("Confusion Metrics", fontsize = 22)
plt.tight_layout()
plt.savefig("confusion_metrics.jpg",dpi=130)
plt.close()

def compute_feature_importance(voting_clf, weights):
    """ Function to compute feature importance of Voting Classifier """

    feature_importance = dict()
    for est in voting_clf.estimators_:
        feature_importance[str(est)] = est.feature_importances_

    fe_scores = [0]*len(list(feature_importance.values())[0])
    for idx, imp_score in enumerate(feature_importance.values()):
        imp_score_with_weight = imp_score*weights[idx]
        fe_scores = list(np.add(fe_scores, list(imp_score_with_weight)))
    return fe_scores

df3 = pd.DataFrame(columns = ["Feature","Feature Importance"])
df3["Feature"] = X_train0.columns
df3["Feature Importance"] = compute_feature_importance(eclf1,[1,1,1])
df3 = df3.sort_values("Feature Importance",ascending = False)

axis_fs = 18 #fontsize
title_fs = 22 #fontsize
sns.set(style="whitegrid")

ax = sns.barplot(x="Feature Importance", y="Feature", data=df3)
ax.set_xlabel('Feature Importance',fontsize = axis_fs) # xlabel
ax.set_ylabel('Feature', fontsize = axis_fs) # ylabel
ax.set_title('Voting Classifier\nFeature Importance', fontsize = title_fs)

plt.tight_layout()
plt.savefig("feature_importance.jpg",dpi=120)
plt.close()

with open("metrics.json", 'w') as outfile:
    json.dump({ "accuracy": acc, "specificity": specificity, "sensitivity":sensitivity}, outfile)

with open("metrics.txt", 'w') as f:
    acc = repr(acc)
    specificity = repr(specificity)
    sensitivity = repr(sensitivity)
    f.write("Accuracy: "+ acc)
    f.write("\nSpecificity: "+ specificity)
    f.write("\nSensitivity: "+ sensitivity)
