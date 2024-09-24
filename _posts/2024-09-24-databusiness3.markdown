---
layout: post
title: "[파이썬 데이터분석 실무 테크닉 100] 03장. 고객의 전체 모습을 파악하는 테크닉 10"
date: 2023-09-24
categories: "데이터비지니스"
---
- use_log.csv: 센터 이용 이력 (2018.4 ~ 2019.3)
- customer_master.csv: 회원 데이터 (2019.3 시점, 탈퇴 회원 포함)
- class_master.csv: 회원 구분 (종일, 주간, 야간)
- campaign_master.csv: 가입 시 행사 종류


### **21. 데이터를 읽어서 파악하자.**

```python
import pandas as pd
uselog = pd.read_csv('use_log.csv')
print(len(uselog))
uselog.head()


customer = pd.read_csv('customer_master.csv')
print(len(customer))
customer.head()


class_master = pd.read_csv('class_master.csv')
print(len(class_master))
class_master.head()

campaign_master = pd.read_csv('campaign_master.csv')
print(len(campaign_master))
campaign_master.head()
```
-> len(): 데이터 개수 파악    
-> 고객 데이터를 메인으로 진행 (uselog 보다 데이터 수가 적음.)

### **22. 고객 데이터를 가공하자.**

```python
customer_join = pd.merge(customer, class_master, on="class", how="left")
customer_join = pd.merge(customer_join, campaign_master, on="campaign_id", how="left")
customer_join.head()
```
customer에 class_master(회원구분)과 campaign_master(캠페인 구분)을 결합하여 customer_join() 생성   
-> 고객 데이터를 중심으로 가로로 결합하는 레프트 조인

#### # 조인 전후 데이터 개수 확인

```python
print(len(customer))
print(len(customer_join))
```
#### # 결측치 확인

```python
customer_join.isnull().sum()
```
-> end_date 외에는 결측치가 0   
=> 탈퇴하지 않은 회원의 탈퇴일이 공백이기 때문.

### **23. 고객 데이터를 집계하자.**

```python
customer_join.groupby("class_name").count()["customer_id"]

customer_join.groupby("campaign_name").count()["customer_id"]

customer_join.groupby("gender").count()["customer_id"]

customer_join.groupby("is_deleted").count()["customer_id"]
```
#### # 특정 기간에 가입한 인원 집계

```python
customer_join["start_date"] = pd.to_datetime(customer_join["start_date"])
 # datetime형으로 변환
customer_start = customer_join.loc[customer_join["start_date"]>pd.to_datetime("20180401")] 
print(len(customer_start))
```
-> 2028.4.1 ~ 2019.3.31에 가입한 인원 집계

### **24. 최신 고객 데이터를 집계하자.**

```python
customer_join["end_date"] = pd.to_datetime(customer_join["end_date"])
customer_newer = customer_join.loc[(customer_join["end_date"]>=pd.to_datetime("20190331"))|(customer_join["end_date"].isna())]
print(len(customer_newer)) # 출력: 2953
customer_newer["end_date"].unique() # 출력: array(['NaT', '2019-03-31T00:00:00.000000000'], dtype='datetime64[ns]')
```
-> 2019.3.31에 탈퇴한 고객과 재적 중인 고객 추출

#### # 회원 구분, 캠페인 구분, 성별 별로 최신 고객 집계

```python
customer_newer.groupby("class_name").count()["customer_id"]

customer_newer.groupby("campaign_name").count()["customer_id"]

customer_newer.groupby("gender").count()["customer_id"]
```

### **25. 이용 이력 데이터를 집계하자.**

#### # 고객마다 월 이용 횟수 집계

```python
uselog["usedate"] = pd.to_datetime(uselog["usedate"])
uselog["연월"] = uselog["usedate"].dt.strftime("%Y%m") # 연월 칼럼 데이터 작성 (ex. 201804)
uselog_months = uselog.groupby(["연월","customer_id"],as_index=False).count()  # 연월과 고객ID 별로 집계
uselog_months.rename(columns={"log_id":"count"}, inplace=True)
del uselog_months["usedate"] # 필요없는 usedate 삭제
uselog_months.head()
```
#### # 고객별로 평균값, 중앙값, 최댓값, 최솟값 집계

```python
uselog_customer = uselog_months.groupby("customer_id").agg(["mean", "median", "max", "min" ])["count"]
uselog_customer = uselog_customer.reset_index(drop=False) 
uselog_customer.head()
```
-> 2행에서 groupby의 영향으로 customer_id가 index에 들어 있기 때문에 이를 칼럼으로 변경.


### **26. 이용 이력 데이터로부터 정기 이용 플래그를 작성하자.**

#### # 고객마다 월/요일별로 집계
```python
uselog["weekday"] = uselog["usedate"].dt.weekday # 요일을 숫자로 변환 (월~일 -> 0~6)
uselog_weekday = uselog.groupby(["customer_id","연월","weekday"], 
                                as_index=False).count()[["customer_id","연월", "weekday","log_id"]] # 고객, 연월, 요일별로 log_id count
uselog_weekday.rename(columns={"log_id":"count"}, inplace=True)
uselog_weekday.head()
```

#### # 고객별로 최댓값 계산, 플래그 지정
최댓값이 4 이상인 요일이 하나라도 있는 회원은 플래그를 1로 처리

```python
uselog_weekday = uselog_weekday.groupby("customer_id",as_index=False).max()[["customer_id", "count"]] # 고객별로 집계, 최댓값 계산
uselog_weekday["routine_flg"] = 0
uselog_weekday["routine_flg"] = uselog_weekday["routine_flg"].where(uselog_weekday["count"]<4, 1) # 4 이상인 경우에만 routine_flg에 1 대입
uselog_weekday.head()
```
특정 월, 특정 요일에 가장 많이 이용한 회수가 4 또는 5 -> 어떤 달의 매주 특정 요일에 정기적으로 방문


### **27. 고객 데이터와 이용 이력 데이터를 결합하자.**
uselog_customer, uselog_weekday, customer_join 결합

```python 
customer_join = pd.merge(customer_join, uselog_customer, on="customer_id", how="left") 
customer_join = pd.merge(customer_join, uselog_weekday[["customer_id", "routine_flg"]], on="customer_id", how="left") 
customer_join.head()
```

#### # 결측치 확인
```python
customer_join.isnull().sum()
```


### **28. 회원 기간을 계산하자.**
아직 탈퇴하지 않은 회원은 end_date를 2019.4.30으로 채워서 기간 계산 (2019.3.31에 탈퇴한 사람과 구분하기 위함.)

```python
from dateutil.relativedelta import relativedelta

customer_join["calc_date"] = customer_join["end_date"] # clac_date 칼럼: 날짜 계산용 칼럼
customer_join["calc_date"] = customer_join["calc_date"].fillna(pd.to_datetime("20190430"))

customer_join["membership_period"] = 0

for i in range(len(customer_join)):
    delta = relativedelta(customer_join["calc_date"].iloc[i], customer_join["start_date"].iloc[i])
    customer_join["membership_period"].iloc[i] = delta.years*12 + delta.months # 회원 기간을 월 단위로 계산

customer_join.head()
```
-> relativedelta(): 날짜 비교 함수

### **29. 고객 행동의 각종 통계량을 파악하자.**

```python
customer_join[["mean", "median", "max", "min"]].describe()
```
```python
customer_join.groupby("routine_flg").count()["customer_id"] 
# 출력: 0 799 / 1 3413
```
-> 정기적으로 이용하는 회원의 수가 많음.

#### # 회원 기간의 분포 확인 

```python
import matplotlib.pyplot as plt
%matplotlib inline
plt.hist(customer_join["membership_period"])
```
-> 회원 기간이 10개월 이내인 고객이 많고, 10개월 이상의 고객 수는 거의 일정.   
=> 짧은 기간에 고객이 빠져나가는 업계임을 시사.


### **30. 탈퇴 회원과 지속 회원의 차이를 파악하자.**

```python
customer_end = customer_join.loc[customer_join["is_deleted"]==1]
customer_end.describe()

customer_stay = customer_join.loc[customer_join["is_deleted"]==0]
customer_stay.describe()
```

#### # join 데이터셋 csv로 출력

```python
customer_join.to_csv("customer_join.csv", index=False)
```