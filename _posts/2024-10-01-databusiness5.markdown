---
layout: post
title: "[파이썬 데이터분석 실무 테크닉 100] 05장. 회원 탈퇴를 예측하는 테크닉 10"
date: 2023-10-01
categories: "데이터비지니스"
---
- use_log.csv: 센터 이용 이력 (2018.4 ~ 2019.3)
- customer_master.csv: 회원 데이터 (2019.3 시점, 탈퇴 회원 포함)
- class_master.csv: 회원 구분 (종일, 주간, 야간)
- campaign_master.csv: 가입 시 행사 종류
- customer_join.csv: 3장에서 작성한 이용 이력을 포함한 고객 데이터
- use_log_months.csv: 4장에서 작성한 이용 이력을 연월/고객별로 집계한 데이터


### **41. 데이터를 읽어 들이고 이용 데이터를 수정하자.**

##### # 데이터 불러오기

```python
import pandas as pd
customer = pd.read_csv('customer_join.csv')
uselog_months = pd.read_csv('use_log_months.csv')
```
#### # 이달과 1개월 전의 이용 횟수 집계

```python 
year_months = list(uselog_months["연월"].unique()) # 연월 칼럼 리스트화
uselog = pd.DataFrame()
for i in range(1, len(year_months)):
    tmp = uselog_months.loc[uselog_months["연월"]==year_months[i]]
    tmp.rename(columns={"count":"count_0"}, inplace=True)
    tmp_before = uselog_months.loc[uselog_months["연월"]==year_months[i-1]]
    del tmp_before["연월"]
    tmp_before.rename(columns={"count":"count_1"}, inplace=True)
    tmp = pd.merge(tmp, tmp_before, on="customer_id", how="left")
    uselog = pd.concat([uselog, tmp], ignore_index=True)
uselog.head()
```

### **42. 탈퇴 전월의 탈퇴 고객 데이터를 작성하자.**

```python

from dateutil.relativedelta import relativedelta
exit_customer = customer.loc[customer["is_deleted"]==1]
exit_customer["exit_date"] = None
exit_customer["end_date"] = pd.to_datetime(exit_customer["end_date"])

for i in range(len(exit_customer)):
    exit_customer["exit_date"].iloc[i] = exit_customer["end_date"].iloc[i] - relativedelta(months=1) # end_date 1개월 전 계산
exit_customer["연월"] = pd.to_datetime(exit_customer["exit_date"]).dt.strftime("%Y%m")
uselog["연월"] = uselog["연월"].astype(str)
exit_uselog = pd.merge(uselog, exit_customer, on=["customer_id", "연월"], how="left")
print(len(uselog))
exit_uselog.head()
```
#### # 결측치가 없는 데이터만 남김

```python
exit_uselog = exit_uselog.dropna(subset=["name"])
print(len(exit_uselog))
print(len(exit_uselog["customer_id"].unique()))
exit_uselog.head()
```

### **43. 지속 회원의 데이터를 작성하자.**

#### # 지속 회원 추출하여 uselog 데이터에 결합

```python
conti_customer = customer.loc[customer["is_deleted"]==0]
conti_uselog = pd.merge(uselog, conti_customer, on=["customer_id"], how="left")
print(len(conti_uselog))
conti_uselog = conti_uselog.dropna(subset=["name"]) # name 칼럼의 결손 데이터 제거
print(len(conti_uselog)) 
```
#### # 데이터를 섞고 중복 제거

```python
conti_uselog = conti_uselog.sample(frac=1).reset_index(drop=True) # 데이터를 섞음.
conti_uselog = conti_uselog.drop_duplicates(subset="customer_id") # customer_id가 중복될 경우 처음 데이터만 가져옴,
print(len(conti_uselog))
conti_uselog.head()
```
#### # 지속 회원 데이터와 탈퇴 회원 데이터 세로로 결합

```python
predict_data = pd.concat([conti_uselog, exit_uselog],ignore_index=True)
print(len(predict_data))
predict_data.head()
```

### **44. 예측할 달의 재적 기간을 작성하자.**

#### # 재적 기간 열 추가

```python
predict_data["period"] = 0
predict_data["now_date"] = pd.to_datetime(predict_data["연월"], format="%Y%m")
predict_data["start_date"] = pd.to_datetime(predict_data["start_date"])
for i in range(len(predict_data)):
    delta = relativedelta(predict_data["now_date"][i], predict_data["start_date"][i])
    predict_data["period"][i] = int(delta.years*12 + delta.months)
predict_data.head()
```

### **45. 결측치를 제거하자.**

#### # 결측치 수 파악
```python
predict_data.isna().sum() 
```
#### # count_1이 결손된 데이터 제거
```python
predict_data = predict_data.dropna(subset=["count_1"]) 
predict_data.isna().sum() 
```

### **46. 문자열 변수를 처리할 수 있게 가공하자.**

- 카테고리 변수: 문자열 데이터 ex) 성별, ~구분
- 더미 변수

```python
target_col = ["campaign_name", "class_name", "gender", "count_1", "routine_flg", "period", "is_deleted"]
predict_data = predict_data[target_col]
predict_data.head()
```

#### # 더미 변수화

```python
predict_data = pd.get_dummies(predict_data)
predict_data.head()

del predict_data["campaign_name_일반"]  
del predict_data["class_name_야간"]
del predict_data["gender_M"] # gender F가 아니면 M이니까.
predict_data.head()
```

### **47. 의사결정 트리를 사용해 탈퇴 예측 모델을 구축하자.**

```python
from sklearn.tree import DecisionTreeClassifier
import sklearn.model_selection
```
#### # 탈퇴, 유지 데이터의 개수 비율 맞추기

```python
exit = predict_data.loc[predict_data["is_deleted"]==1]
conti = predict_data.loc[predict_data["is_deleted"]==0].sample(len(exit))
```
#### # 모델 학습

```python
X = pd.concat([exit, conti], ignore_index=True)
y = X["is_deleted"] # 목적 변수 y에 저장
del X["is_deleted"] # X에서 제거
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y) # 학습, 평가 데이터로 나누기

model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
print(y_test_pred)

results_test = pd.DataFrame({"y_test":y_test ,"y_pred":y_test_pred })
results_test.head()
```

### **48. 예측 모델을 평가하고, 모델을 튜닝하자.**

#### # 모델 평가

```python
correct = len(results_test.loc[results_test["y_test"]==results_test["y_pred"]])
data_count = len(results_test)
score_test = correct / data_count # 정답률
print(score_test) # 출력: 0.91

print(model.score(X_test, y_test)) # 출력: 0.91...
print(model.score(X_train, y_train)) # 출력: 0.98...
```
-> 과적합
#### # 모델 파라미터 변경

```python
X = pd.concat([exit, conti], ignore_index=True)
y = X["is_deleted"]
del X["is_deleted"]
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y)

model = DecisionTreeClassifier(random_state=0, max_depth=5)
model.fit(X_train, y_train)
print(model.score(X_test, y_test)) # 출력: 0.92...
print(model.score(X_train, y_train)) # 출력: 0.92...
```

### **49. 모델에 기여하고 있는 변수를 확인하자.**

```python
importance = pd.DataFrame({"feature_names":X.columns, "coefficient":model.feature_importances_})
importance
```
-> model.feature_importances_: 변수 중요도 출력

### **50. 회원의 탈퇴를 예측하자.**

#### # 예시 데이터 작성
```python
count_1 = 3
routing_flg = 1
period = 10
campaign_name = "입회비무료"
class_name = "종일"
gender = "M"
```

#### # 예측
```python
if campaign_name == "입회비반값할인":
    campaign_name_list = [1, 0]
elif campaign_name == "입회비무료":
    campaign_name_list = [0, 1]
elif campaign_name == "일반":
    campaign_name_list = [0, 0]
if class_name == "종일":
    class_name_list = [1, 0]
elif class_name == "주간":
    class_name_list = [0, 1]
elif class_name == "야간":
    class_name_list = [0, 0]
if gender == "F":
    gender_list = [1]
elif gender == "M":
    gender_list = [0]

input_data = [count_1, routing_flg, period]
input_data.extend(campaign_name_list)
input_data.extend(class_name_list)
input_data.extend(gender_list)

print(model.predict([input_data])) # 출력: [1. ]
print(model.predict_proba([input_data])) # 출력: [0.01...  0.98... ]
```



