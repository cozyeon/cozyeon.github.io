---
layout: post
title: "[파이썬 데이터분석 실무 테크닉 100] 04장. 고객의 행동을 예측하는 테크닉 10"
date: 2023-09-24
categories: "데이터비지니스"
---
- use_log.csv: 센터 이용 이력 (2018.4 ~ 2019.3)
- customer_master.csv: 회원 데이터 (2019.3 시점, 탈퇴 회원 포함)
- class_master.csv: 회원 구분 (종일, 주간, 야간)
- campaign_master.csv: 가입 시 행사 종류
- customer_join.csv: 3장에서 작성한 이용 이력을 포함한 고객 데이터


### **31. 데이터를 읽어 들이고 확인하자.**

```python
import pandas as pd

uselog = pd.read_csv('use_log.csv')
uselog.isnull().sum()

customer = pd.read_csv('customer_join.csv')
customer.isnull().sum()
```

### **32. 클러스터링으로 회원을 그룹화하자.**
이용 이력을 이용해서 그룹화

#### # 필요한 변수 추출
```python
customer_clustering = customer[["mean", "median","max", "min", "membership_period"]]
customer_clustering.head()
```

#### # K-means로 그룹화

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
customer_clustering_sc = sc.fit_transform(customer_clustering) # 표준화

kmeans = KMeans(n_clusters=4, random_state=0) # 클러스터 수 4로 지정
clusters = kmeans.fit(customer_clustering_sc) 
customer_clustering["cluster"] = clusters.labels_ # 원래 데이터에 결과 반영

print(customer_clustering["cluster"].unique())
customer_clustering.head()
```


### **33. 클러스터링 결과를 분석하자.**

```python
customer_clustering.columns = ["월평균값","월중앙값", "월최댓값", "월최솟값","회원기간", "cluster"] # 칼럼 이름 변경

customer_clustering.groupby("cluster").count() # 클러스터마다 집계

customer_clustering.groupby("cluster").mean() # 그룹마다 평균값 계산
```


### **34. 클러스터링 결과를 가시화하자.**

**차원축소:** 비지도 학습의 일종, 정보를 되도록 잃지 않게 하면서 새로운 축을 만드는 것    
-> 대표적 방법: 주성분 분석

#### # 주성분 분석
```python
from sklearn.decomposition import PCA
X = customer_clustering_sc
pca = PCA(n_components=2) # 모델 생성
pca.fit(X) 
x_pca = pca.transform(X)
pca_df = pd.DataFrame(x_pca)
pca_df["cluster"] = customer_clustering["cluster"]
```

#### # 가시화 
```python
import matplotlib.pyplot as plt
%matplotlib inline
for i in customer_clustering["cluster"].unique(): # 그룹마다 색상 바꿈
    tmp = pca_df.loc[pca_df["cluster"]==i]
    plt.scatter(tmp[0], tmp[1])
```

### **35. 클러스터링 결과를 바탕으로 탈퇴 회원의 경향을 파악하자.**

```python
ustomer_clustering = pd.concat([customer_clustering, customer], axis=1)

customer_clustering.groupby(["cluster","is_deleted"],as_index=False).count()[["cluster","is_deleted","customer_id"]]
# cluster, is_deleted 별로 customer_id의 개수 집계
```
```python
customer_clustering.groupby(["cluster","routine_flg"],as_index=False).count()[["cluster","routine_flg","customer_id"]]
# cluster, routine_flg 별로 customer_id의 건수 집계
```

### **36. 다음 달의 이용 횟수 예측을 위해 데이터를 준비하자.**
2018.5 ~ 2018.10의 데이터로 2018.11의 이용 횟수 예측

#### # 연월, 고객별로 log_id 집계
```python
uselog["usedate"] = pd.to_datetime(uselog["usedate"])
uselog["연월"] = uselog["usedate"].dt.strftime("%Y%m")
uselog_months = uselog.groupby(["연월","customer_id"],as_index=False).count() 
uselog_months.rename(columns={"log_id":"count"}, inplace=True) 
del uselog_months["usedate"]
uselog_months.head()
```
#### #
```python
year_months = list(uselog_months["연월"].unique()) # 연월 데이터를 리스트에 저장
predict_data = pd.DataFrame()
for i in range(6, len(year_months)): # 2018.10 ~ 2019.3 데이터 취득해서 저장
    tmp = uselog_months.loc[uselog_months["연월"]==year_months[i]]
    tmp.rename(columns={"count":"count_pred"}, inplace=True)
    for j in range(1, 7):
        tmp_before = uselog_months.loc[uselog_months["연월"]==year_months[i-j]]
        del tmp_before["연월"]
        tmp_before.rename(columns={"count":"count_{}".format(j-1)}, inplace=True)
        tmp = pd.merge(tmp, tmp_before, on="customer_id", how="left")
    predict_data = pd.concat([predict_data, tmp], ignore_index=True)
predict_data.head()
```
-> count_pred 컬럼: 예측하고 싶은 달의 데이터

#### # 결측치 처리
```
predict_data = predict_data.dropna() # 결측치를 포함하는 데이터 삭제
predict_data = predict_data.reset_index(drop=True) # 인덱스 초기화
predict_data.head()
```

### **37. 특징이 되는 변수를 추가하자.**
회원 기간을 추가.

#### # start_date 칼럼 predict_data에 결합

```python
predict_data = pd.merge(predict_data, customer[["customer_id","start_date"]], on="customer_id", how="left") 
predict_data.head()
```

#### # 회원 기간을 월 단위로 작성

```python
predict_data["now_date"] = pd.to_datetime(predict_data["연월"], format="%Y%m") # 연월 칼럼 datetime형으로 변환
predict_data["start_date"] = pd.to_datetime(predict_data["start_date"]) # start_date 칼럼 datetime형으로 변환

from dateutil.relativedelta import relativedelta 
predict_data["period"] = None
for i in range(len(predict_data)): # 회원 기간 계산
    delta = relativedelta(predict_data["now_date"][i], predict_data["start_date"][i])
    predict_data["period"][i] = delta.years*12 + delta.months
predict_data.head()
```


### **38. 다음 달 이용 횟수를 예측하는 모델을 구축하자.**
오랜 회원은 제외하고 2018.4~ 회원 데이터만 이용.

```python
predict_data = predict_data.loc[predict_data["start_date"]>=pd.to_datetime("20180401")] # 2018.4 이후 회원만 추출

from sklearn import linear_model
import sklearn.model_selection

model = linear_model.LinearRegression() # 모델 초기화
X = predict_data[["count_0","count_1","count_2","count_3","count_4","count_5","period"]] # 변수 x 정의
y = predict_data["count_pred"] # 변수 y 정의
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y) # 학습용 데이터와 평가용 데이터로 분할
model.fit(X_train, y_train) # 모델 훈련

print(model.score(X_train, y_train)) # 정확도 검증
print(model.score(X_test, y_test))
```
### **39. 모델에 기여하는 변수를 확인하자.**

```python
coef = pd.DataFrame({"feature_names":X.columns, "coefficient":model.coef_})
coef
```
-> 과거로 거슬러 올라갈수록 기여도 감소   
=> 이전 다르이 이용 횟수가 다음 달의 이용횟수에 영향을 미치고 있음.


### **40. 다음달의 이용횟수를 예측하자.**
```python

x1 = [3, 4, 4, 6, 8, 7, 8] # 회원1
x2 = [2, 2, 3, 3, 4, 6, 8] # 회원2
x_pred = [x1, x2]

model.predict(x_pred) # 예측 진행

uselog_months.to_csv("use_log_months.csv",index=False)
```