## Confusion Matrix (오차행렬)

🌟 오차행렬이란?: **test data set에 대한 분류기(classifier) 즉, 분류의 성능을 평가하는 행렬**

- 학습된 분류 모델이 예측을 수행하면서 얼마나 헷갈리고 있는지도 함께 보여주는 지표. 어떠한 유형의 예측 오류가 발생하는지 확인할 수 있다



```python
# real = 실제 값,   prediction = 예측한 값
from sklearn.metrics import confusion_matrix
confusion_matrix(real, prediction)  #confusion matrix 표시


from sklearn.metrics import precision_score, recall_score, f1_score
precision_score(real, prediction)  # precision(정밀도)
recall_score(real, prediction)     # recall(재현율), sensitivity(민감도), 
f1_score(real, prediction)         # F1 score(정밀도, 민감도 조화평균)
```

<br></br>

Confusion matrix는 아래와 같은 형식을 따른다:

> Row: 실제 class
>
> Column: 예측한 class

|                | Predicted NO   | Predicted YES  |
| -------------- | -------------- | -------------- |
| **Actual NO**  | True Negative  | False Positive |
| **Actual YES** | False Negative | True Positive  |



-> 이해하기 쉬운 예시!

🌟 **상황 가정**: *비가 올지 말지 몰라서 우산을 챙길지 말지 정하는 상황!*

1. 실제로 비가 온다 (True) vs 비가 오지 않는다 (False)
2. 우산을 챙긴다 (Positive) vs 챙기지 않는다 (Negative)



|                                         | Negative (우산을 안 챙긴다) | Positive (우산을 챙긴다) |
| --------------------------------------- | --------------------------- | ------------------------ |
| **Negative**  (예측과 실제 값이 불일치) | True Negative               | False Positive           |
| **Positive** (예측과 실제 값이 일치)    | False Negative              | True Positive            |

** 헷갈리지 말아야 할 것: *True, false는 실제값과 예측값이 일치하는지 판별하는 것이다*

1. 우산을 챙겼는데 (P) 비가 왔다면(T)? : Positive가 True! -> 예상 적중!
2. 우산을 챙겼는데 (P) 비가 안 왔다면(T)? Positive가 False -> 짐만 되는 우산을 들고 나왔다 ㅠㅠ
3. 우산을 안 챙겼는데 (N) 비가 왔다면(F)?: Negative가 False -> 비를 맞고 다니거나 우산을 사야한다 ㅠㅠ
4. 우산을 안 챙겼는데 (N) 비가 안 왔다면(T)? Negative가 True -> 눈치게임 성공!



여기서 좋다고 판단되는 경우는 1번과 4번이다! 따라서 1번과 4번의 비율이 높을수록 이상적인 결과이다. 특히나 True Positive의 비율이 높을 수록 예측을 잘 했다고 판단된다. 



> 코드로 보는 예시

```python
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, pred)

# 133명 중, 여학생이 10명인 케이스에 대해 모두 남자라고 예측한다면 다음과 같은 결과가 나온다.
array([[126, 0],    
    [10,0]], dtype=int64)
# 133명을 남자라고 예측하고 실제 남자인 경우 126명 (True Negative)
# 133명을 남자라고 예측하고 실제 남자인 경우 0 명 (True Negative)
```

<br></br>

## Precision (정밀도) 와 Recall (재현율)

🌟 불균형 데이터셋에서는 **정밀도(Precision)** 와 **재현율(Recall)** 을 많이 사용한다. 예측 성능에 좀 더 초점을 맞춘 평가 지표이다. 

- <u>정확도</u>는 Positive로 예측한 값들 중에 실제로 Positive한 값의 비율
  - 위의 예시에서 내가 <u>비가 온다고 예상했을 때</u>(True) <u>우산을 가지고 온</u>(Positive) 비율을 말한다.
- <u>재현율</u>은 실제 값이 Positive인 값들 중에 예측을 Positive로 한 값의 비율
  - 위의 예시에서 <u>실제로 비가 올 때(True)</u>, <u>우산을 잘 챙기느냐(P)</u>를 평가하는 방법

```python
from sklearn.metrics import accuracy_score, precision_score

print(precision_score(y_train, y_pred))
print(recall_score(y_train, y_pred))
```

**Output**:

precision_score: 0.7788

- 우산을 챙겼을 때 0.77%로 정확히 예상함 (비가 옴)

recall_score: 0.8872

- 전체 데이터 셋에서 우산을 챙긴 비율이 88%

<br></br>

🌟 정밀도와 재현율이 쓰이는 경우 **예시** 

여러가지 업무 케이스에 따라 False Negative, False Positive 일 때의 치명률이 달라진다!

**정밀도는 FP를 낮추는데에 초점**을 두고 있고, **재현율은 FN을 낮추는데에 초점**을 둔다. 

Case 1: 암 판단 모델에서는 암환자에게 (False) 암이 아니라고(Negative) 예측한 경우가 최악의 케이스다. 

- 따라서 실제 암인 환자 중에서 암을 예측한 확률을 구한다 (재현율)

Case 2: 스팸 메일 구분 모델에서는 정상메일(False)을 스팸으로(Positive) 예측한 경우가 최악의 케이스다.

- 따라서 스팸으로 예측한 확률 중, 실제 스팸인 확률을 구한다 (정밀도)

<br></br>

#### 정밀도와 재현율의 Trade-off

정밀도가 늘어나면 재현율은 줄어들고, 정밀도가 줄어들면 재현율은 늘어난다. 

분류하려는 업무의 특성에 따라 재현율을 늘이거나, 정밀도를 늘이게 되는데 이를 조정하기 위해서는 임곗값(threshold)의 정밀한 조정이 필요하다

- F1 Score: 이 정밀도와 재현율을 둘 다 고려한 분류 평가 수치 (정밀도와 재현율의 조화평균)
  - 한 쪽으로 치우치지 않을 때 상대적으로 높은 값을 지님

