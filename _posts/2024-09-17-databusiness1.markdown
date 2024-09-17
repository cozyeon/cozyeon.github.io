---
layout: post
title: "[파이썬 데이터분석 실무 테크닉 100] 01장. 웹에서 주문 수를 분석하는 테크닉 10"
date: 2023-09-17
categories: "데이터 비지니스"
---


### **1. 데이터를 읽어 들이자**
```python
import pandas as pd
customer_master = pd.read_csv('customer_master.csv')
customer_master.head()

item_master=pd.read_csv('item_master.csv')
item_master.head()

item_master=pd.read_csv('transaction1.csv')
item_master.head()

item_master=pd.read_csv('transaction_detail_1.csv')
item_master.head()
```
### **2. 데이터를 결합(유니언)해 보자**

```python
transaction_2 = pd.read_csv('transaction_2.csv')
transaction = pd.concat([transaction1, transaction_2], ignore_index=True)
transcation.head()

transaction_detail_2 = pd.read_csv('transaction_detail_2.csv')
transaction_detail = pd.concat([transaction_detail_1, transaction_detail_2],ignore_index=True)
transaction_detail.head()
```

> pd.concat
>-
> 데이터 유니언 (데이터를 행방향으로 결합)

### **3. 매출 데이터끼리 결합(조인)해 보자**
1. 부족한(추가하고 싶은) 데이터 칼럼 결정
2. 공통되는 데이터 칼럼 고려



```python
join_data=pd.merge(transaction_detail, transaction[["transaction_id", "payment_data", "customer_id"]], on="transaction_id", how="left")
join_data.head()
```

> pd.merge   
>-
> - 인수: 기준, 필요한 칼럼, 조인키(on), 조인 종류(how) 
> - 조인할 데이터의 조인키에 중복 데이터가 존재하는 경우 기준 데이터의 개수가 늘어날 수 있음.


### **4. 마스터데이터를 결합(조인)해 보자**
```python
join_data = pd.merge(join_data, customer_master, on="customer_id", how="left")
join_data = pd.merge(join_data, item_master, on="item_id", how="left")
join_data.head()
```
### **5. 필요한 데이터 칼럼을 만들자**

```python
join_data["price"] = join_data["quantity"]*join_data["item_price"]
join_data[["quantity", "item_price", "price"]].head()
```
-> 데이터프레임형 곱셈: 행마다 계산 실행

### **6. 데이터를 검산하자**

```python
print(join_data["price"].sum())
print(transaction["price"].sum())
```
or
```python
join_data["price"].sum() == transaction["price"].sum()
```
### **7. 각종 통계랑을 파악하자**
1. 결손치의 개수 파악
2. 전체를 파악할 수 있는 숫자감 파악

#### # 결손치 파악
```python
join_data.isnull().sum()
```
-> isnull() : 결손치가 True/False로 반환   
-> isnull().sum() : 컬럼별로 True의 개수를 계산

#### # 통계량 출력
>describe()
>-
>- 전체적인 느낌을 파악하기 위한 각종 통계량을 출력
>- 컬럼별로 데이터 개수, 평균값, 표준편차, 최솟값, 사분위수, 중앙값, 최댓값 출력
```python
join_data.describe()
```

#### # 데이터의 기간 파악
```python
print(join_data["payment_date"].min())
print(join_data["payment_date"].max())
```

### **8. 월별로 데이터를 집계해 보자**

#### # 데이터형 확인
```python
join_data.dtypes
```
#### # 데이터형 변환
```python
join_data["payment_date"] = pd.to_datetime(join_data["payment_date"])
join_data["payment_month"] = join_data["payment_date"].dt.strftime("%Y%m")
join_data[["payment_date", "payment_month"]].head()
```
-> datetime형으로 변환하고, 연월 단위로 작성

> 판다스 datetime의 dt
>-
> 년, 월 추출

#### # 집계
```python
join_data.groupby("payment_month").sum()["price"]
```
-> 월별 매출 표시   
-> payment_month 집계, 합산, price 컬럼으로 표시

> groupby
>-
> 집계하고 싶은 칼럼, 집계 방법, 표시할 컬럼 기술

### **9. 월별, 상품별로 데이터를 집계해 보자**
```python
join_data.groupby(["paymetn_month", "item_name"]).sum()[["price", "quantity"]]
```
-> groupby에서 출력하고 싶은 칼럼이 여러개일 경우 리스트형으로 지정

```python
pd.pivot_table(join_data, index='item_name', columns='payment_month', values=['price', 'quantity'], aggfunc='sum')
```
> pivot_table
>-
>- 행과 칼럼 지정 가능
>- index='행', columns='칼럼', values=['집계하고싶은 칼럼'], aggfunc='집계방법'

### **10. 상품별 매출 추이를 가시화해 보자**


```python
graph_data = pd.pivot_table(join_data, index='payment_month', columns='item_name', values='price', aggfunc='sum')
graph_data.head()
```
```python
import matplotlib.pyplot as plt
%matplotlib inline # 주피터노트북에 표시
plt.plot(list(graph_data.index), graph_data["PC-A"], label='PC-A')
plt.plot(list(graph_data.index), graph_data["PC-B"], label='PC-B')
plt.plot(list(graph_data.index), graph_data["PC-C"], label='PC-C')
plt.plot(list(graph_data.index), graph_data["PC-D"], label='PC-D')
plt.plot(list(graph_data.index), graph_data["PC-E"], label='PC-E')
plt.legend()  # 범례 표시
``` 