import numpy as np
import pdb

from mlquantify.adjust_counting import GAC, GPAC
from mlquantify.neighbors import KDEyHD
from mlquantify.likelihood import EMQ
from sklearn.linear_model import LogisticRegression



if __name__ == "__main__":
    # emq = EMQ()
    # emq.fit()

    # emq.priors = [0.3, 0.7]

    gac = GAC(learner=LogisticRegression(n_jobs=-1))
    X = np.random.randn(50, 4)
    y = np.random.randint(0, 3, 50)
    # pdb.set_trace()
    gac.fit(X, y)
    print(gac.predict(X))

