import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)


df = pd.read_csv('./Data/emails.train.csv')

train_data = pd.read_csv('./Data/emails.train.csv')
test_data  = pd.read_csv('./Data/emails.test.csv')


# fit the model
clf = IsolationForest(max_samples=100, random_state=rng)
clf.fit(train_data)
y_pred_train = clf.predict(test_data)
y_pred_test = clf.predict(test_data)

# plot the line, the samples, and the nearest vectors to the plane
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("IsolationForest")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

b1 = plt.scatter(train_data[:, 0], train_data[:, 1], c='white',
                 s=20, edgecolor='k')
b2 = plt.scatter(train_data[:, 0], train_data[:, 1], c='green',
                 s=20, edgecolor='k')

plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([b1, b2, c],
           ["training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left")
plt.show()