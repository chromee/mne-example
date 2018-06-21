from sklearn import datasets, svm
import numpy as np

iris = datasets.load_iris()

clf = svm.SVC()
clf.fit(iris.data, iris.target)

# setosaの特徴量を与えてちゃんと分類してくれるか試します
test_data = [[ 5.1,  3.5,  1.4,  0.2]]
print(clf.predict(test_data))