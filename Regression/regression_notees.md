## Random Forest

어렵지 않고 사용하기 쉬워 초보자나 현업에서 많이 사용됨

Boosting 계열에 비해 하이퍼 파라미터에 민감하지 않음

Bagging의 문제점을 개선하기 위해 등장한 개념

\

Bagging의 문제점:

- 트리의 상관성이 높다면 앙상블의 의미가 줄어든다.

- Tree의 불순도를 기준으로 분할하게 되는데, 잘 분할하는 변수가 존재한다면 첫 번째 분할은 계속해서 한 변수가 분할할 것이다.
- 이로 인해 tree의 다양성이 낮아지게 된다.

\

Random feature selection을 통한 tree의 다양성 확보

- Bootstrap + Random feature selection을 통해 개별 트리의 다양성을 극대화 시킴

> 사용 예시

```python
from sklearn.ensemble import RandomForestClassifier #분류
from sklearn.ensemble import RandomForestRegressor #회귀

rf1 = RandomForestClassifier(random_state=1203)
rf1.fit(X_train, y_train);

rf2 = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf2.fit(X_train, y_train);
```

\

> [파라미터](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

```markdown
- n_estimators = number of trees in the foreset
- max_features = max number of features considered for splitting a node
- max_depth = max number of levels in each decision tree
- min_samples_split = min number of data points placed in a node before the node is split
- min_samples_leaf = min number of data points allowed in a leaf node
- bootstrap = method for sampling data points (with or without replacement)
```

Cross Validation을 통해 해당 파라미터들을 튜닝하여 최적 파라미터를 찾아 최적의 모델을 찾아낸다.

