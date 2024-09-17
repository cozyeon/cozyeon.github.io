---
layout: post
title: "[파이썬 데이터분석 실무 테크닉 100] 02장. 대리점 데이터를 가공하는 테크닉 10"
date: 2023-09-17
categories: "데이터 비지니스"
---
### **11. 데이터를 읽어들이자**
```python
import pandas as pd

uriage_data = pd.read_csv("uriage.csv")
uriage_data.head()

kokyaku_data = pd.read_excel("kokyaku_daicho.xlsx")
kokyaku_data.head()
```
데이터에 나타나는 입력 오류나 표기 방법의 차이가 부정합을 일으킬 때   
=> 데이터의 정합성에 문제가 있다
ex) 김현성 / 김 현성 / 김  현성/ 김현 성

### **12. 데이터의 오류를 살펴보자**

```python
uriage_data["item_name"].head()
```
-> 공백이 포함되거나 알파벳 대소문자가 섞여 있음.

```
uriage_data["item_price"].head()
```
-> 결측치 NaN 존재

```
kokyaku_data["등록일"].head()

```

### **13. 데이터에 오류가 있는 상태로 집계해 보자**

#### # 매출 이력에서 상품별로 월 판매 개수 집계

```python
uriage_data["purchase_date"] = pd.to_datetime(uriage_data["purchase_date"])
uriage_data["purchase_month"] = uriage_data["purchase_date"].dt.strftime("%Y%m")

res = uriage_data.pivot_table(index="purchase_month", columns="item_name", aggfunc="size", fill_value=0) # 상품의 건수 집계
res
```

#### # 매출 이력에서 상품별로 월 매출 집계 
```python
res = uriage_data.pivot_table(index="purchase_month", columns="item_name", values="item_price", aggfunc="sum", fill_value=0)
res
```
-> 동일한 상품이 다른 상품으로 집계됨. -> 상품 수가 늘어남.

### **14. 상품명 오류를 수정하자**

#### # 중복을 제외한 데이터 건수 확인

```python
print(len(pd.unique(uriage_data["item_name"]))) # 출력: 99
```
-> pd.unique(): 중복 제외한 데이터

#### # 데이터 가공
```python
uriage_data["item_name"] = uriage_data["item_name"].str.upper()
uriage_data["item_name"] = uriage_data["item_name"].str.replace("　", "")
uriage_data["item_name"] = uriage_data["item_name"].str.replace(" ", "")
uriage_data.sort_values(by=["item_name"], ascending=True)
```
-> str.upper(): 소문자를 대문자로 변환   
-> str.replace(): 대체 => 공백 제거   
-> sort_values(): 데이터 정렬

```python
print(pd.unique(uriage_data["item_name"]))
print(len(pd.unique(uriage_data["item_name"]))) # 출력: 26
```

### **15. 금액의 결측치를 수정하자**

#### # 결측치 있는지 확인
```python
uriage_data.isnull().any(axis=0)
```

#### # 결손치 수정

```python
flg_is_null = uriage_data["item_price"].isnull() # 결측치 있는 곳 조사
for trg in list(uriage_data.loc[flg_is_null, "item_name"].unique()):  # 결측치가 있는 상품명 추출
    price = uriage_data.loc[(~flg_is_null) & (uriage_data["item_name"] == trg), "item_price"].max() # 결손치가 있는 상품과 같은 상품의 다른 데이터에서 금액을 가져옴
    uriage_data["item_price"].loc[(flg_is_null) & (uriage_data["item_name"]==trg)] = price # 가져온 금액으로 데이터 수정
uriage_data.head()
```
-> log[조건, 칼럼]: 조건에 일치하는 데이터 중 해당 칼럼에 해당하는 데이터 추출

#### # 결측치 없어졌는지 확인
```python
uriage_data.isnull().any(axis=0)
```

#### # 상품의 금액 정상적으로 수정됐는지 확인
```python
for trg in list(uriage_data["item_name"].sort_values().unique()):
    print(trg + "의최고가：" + str(uriage_data.loc[uriage_data["item_name"]==trg]["item_price"].max()) 
          + "의최저가：" + str(uriage_data.loc[uriage_data["item_name"]==trg]["item_price"].min(skipna=False)))
```
-> skipna: NaN의 무시 여부 설정   
-> 상품에 설정된 최대 금액과 최소 금액 출력   
=> 최대 금액과 최소 금액이 일치하면 한 상품에 하나의 가격이 존재함을 확인 가능   


  
### **16. 고객이름의 오류를 수정하자**

#### # 고객 정보의 고객 이름 확인
```python
kokyaku_data["고객이름"].head()
```
-> 성과 이름 사이 공백

#### # 매출 이력의 고객 이름 확인
```python
uriage_data["customer_name"].head()
```
-> 성과 이름 사이 공백 없음.


#### # 고객 정보의 고객 이름에서 공백 제거
```python
kokyaku_data["고객이름"] = kokyaku_data["고객이름"].str.replace("　", "")
kokyaku_data["고객이름"] = kokyaku_data["고객이름"].str.replace(" ", "")
kokyaku_data["고객이름"].head()
```

### **17. 날짜오류를 수정하자**

```python
flg_is_serial = kokyaku_data["등록일"].astype("str").str.isdigit() 
# 숫자로 된 장소를 fig_is_serial에 저장
flg_is_serial.sum() # 출력: 22
# 검증을 위해 개수 표시
```
-> str.isddigit(): 데이터 타입이 숫자인지 아닌지 판단

#### # 숫자 데이터 날짜로 변환

```python
fromSerial = pd.to_timedelta(kokyaku_data.loc[flg_is_serial, "등록일"].astype("float"), unit="D") + pd.to_datetime("1900/01/01")
fromSerial
```
-> pd.to_timedalta(): 숫자를 날짜로 변환

#### # 날짜 데이터 서식 변경
```python
fromString = pd.to_datetime(kokyaku_data.loc[~flg_is_serial, "등록일"])
fromString
```

#### 데이터 결합
```python
kokyaku_data["등록일"] = pd.concat([fromSerial, fromString])
kokyaku_data
```

#### # 등록월을 추출하여 집계
```python
kokyaku_data["등록연월"] = kokyaku_data["등록일"].dt.strftime("%Y%m")
rslt = kokyaku_data.groupby("등록연월").count()["고객이름"]
print(rslt)
print(len(kokyaku_data)) # 출력: 200
```

#### # 등록일 칼럼에 숫자 데이터가 남아 있는지 확인
```python
flg_is_serial = kokyaku_data["등록일"].astype("str").str.isdigit()
flg_is_serial.sum() # 출력: 0
```

### **18. 고객이름을 키로 두개의 데이터를 결합(조인)하자**
두 개 데이터의 서로 다른 열을 지정하여 결합


```python
join_data = pd.merge(uriage_data, kokyaku_data, left_on="customer_name", right_on="고객이름", how="left")
join_data = join_data.drop("customer_name", axis=1)
join_data
```
-> left_on="customer_name", right_on="고객이름", how="left" : uriage_data를 기준으로 kokyaku_data를 결합

### **19. 정제한 데이터를 덤프하자**
**데이터 정제**: 분석에 적합한 데이터의 형태로 데이터 가공   
**데이터 덤프**: 정제된 데이터를 파일로 출력

#### # 칼럼 배치 조정
```python
dump_data = join_data[["purchase_date", "purchase_month", "item_name", "item_price", "고객이름", "지역", "등록일"]]
dump_data
```

#### # 데이터 덤프
```python
dump_data.to_csv("dump_data.csv", index=False)
```
-> to_csv(): 같은 폴더에 csv 파일로 저장


### **20. 데이터를 집계하자**
```python
import_data = pd.read_csv("dump_data.csv")
import_data
```
#### # purchase_month를 세로축으로 상품별 개수 집계
```python
byItem = import_data.pivot_table(index="purchase_month", columns="item_name", aggfunc="size", fill_value=0)
byItem
```
#### # purchase_month를 세로축으로 상품별 매출 금액 집계
```python
byPrice = import_data.pivot_table(index="purchase_month", columns="item_name", values="item_price", aggfunc="sum", fill_value=0)
byPrice
```
#### # purchase_month를 세로축으로 고객별 개수 집계
```python
byCustomer = import_data.pivot_table(index="purchase_month", columns="고객이름", aggfunc="size", fill_value=0)
byCustomer 
```
#### # purchase_month를 세로축으로 지역별 개수 집계
```python
byRegion = import_data.pivot_table(index="purchase_month", columns="지역", aggfunc="size", fill_value=0)
byRegion
```

#### # 구매 이력이 없는 사용자 확인
```python
away_data = pd.merge(uriage_data, kokyaku_data, left_on="customer_name", right_on="고객이름", how="right")
away_data[away_data["purchase_date"].isnull()][["고객이름", "등록일"]]
```