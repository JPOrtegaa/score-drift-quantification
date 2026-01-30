import pdb
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from mlquantify.adjust_counting import GAC, GPAC, FM
from mlquantify.neighbors import KDEyHD, KDEyCS, KDEyML
from mlquantify.likelihood import EMQ
from mlquantify.mixture import HDx
from mlquantify.neighbors import PWK

from methods.quantifiers import CC2, ACC
from methods.quadapt import ACCSyn
from methods.quantifiers_utils import getTPRandFPRbyThreshold

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Remove instances from class 2
mask = y != 2
X = X[mask]
y = y[mask]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# pdb.set_trace()

# Train the classifier
clf = LogisticRegression(n_jobs=-1, max_iter=200)
clf.fit(X_train, y_train)

priors = clf.predict_proba(X_train)
posteriors = clf.predict_proba(X_test)
# pdb.set_trace()

priors2 = np.column_stack((priors, y_train))

tprfpr = getTPRandFPRbyThreshold(priors2)

train_preds = clf.predict(X_train)  # Use predict() instead of predict_proba()
test_preds = clf.predict(X_test)


gac = GAC()
gpac = GPAC()
emq = EMQ()
kde = KDEyHD()
kde_cs = KDEyCS()
kde_ml = KDEyML()
fm = FM()
hdx = HDx(bins_size=np.linspace(10, 110, 11))
pwk = PWK(n_neighbors=11, n_jobs=-1)

# pdb.set_trace()

gac_r = gac.aggregate(train_predictions=train_preds, predictions=test_preds, y_train_values=y_train)
gpac_r = gpac.aggregate(train_predictions=priors, predictions=posteriors, y_train_values=y_train)
emq_r = emq.aggregate(predictions=posteriors, y_train=y_train)
kde_r = kde.aggregate(train_predictions=priors, predictions=posteriors, train_y_values=y_train)
kde_cs_r = kde_cs.aggregate(train_predictions=priors, predictions=posteriors, train_y_values=y_train)
kde_ml_r = kde_ml.aggregate(train_predictions=priors, predictions=posteriors, train_y_values=y_train)
cc2_r = CC2(test=posteriors)
acc_r = ACC(test=posteriors, TprFpr=tprfpr)
syn_r = ACCSyn(ts=posteriors, MF_dysyn=np.arange(0.1, 1.0, 0.2), measure='hellinger')
fm_r = fm.aggregate(train_predictions=priors, predictions=posteriors, y_train_values=y_train)

hdx.fit(X=X_train, y=y_train)
hdx_r = hdx.predict(X=X_test)
pwk.fit(X=X_train, y=y_train)
pwk_r = pwk.predict(X=X_test)

print("GAC Result:", gac_r)
print("GPAC Result:", gpac_r)
print("EMQ Result:", emq_r)
print("KDEyHD Result:", kde_r)
print("KDEyCS Result:", kde_cs_r)
print("KDEyML Result:", kde_ml_r)
print("CC2 Result:", cc2_r)
print("ACC Result:", acc_r)
print("ACCSyn Result:", syn_r)
print("FM Result:", fm_r)
print("HDx Result:", hdx_r)
print("PWK Result:", pwk_r)