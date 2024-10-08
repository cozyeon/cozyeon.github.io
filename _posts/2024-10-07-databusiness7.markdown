---
layout: post
title: "[파이썬 데이터분석 실무 테크닉 100] 07장. 물류 네트워크 최적 설계를 위한 테크닉 10"
date: 2023-10-07
categories: "데이터비지니스"
---

- tbl.factory.csv: 생산 공장 데이터
- tbl_warehouse.csv: 창고 데이터
- rel_cost.csv: 창고와 공장 간의 운송 비용
- tbl_transaction.csv: 2019년의 공장으로의 부품 운송 실적

- 대리점: 최종적으로 제품 판매
- 제품: 일정 수요가 예측되어 있음.
- 공장: 제품의 예측 수요를 근거로 생산량 결정
- 생산라인: 각 공장에서의 대리점까지의 운송비, 제고 비용 등을 고려하여 어느 생산라인에서 제조할지 결정
![produce_bigpicture](/assets/img/produce_bigpicture.jpg)

### **61. 운송최적화 문제를 풀어보자.**

```python
import numpy as np
import pandas as pd
from itertools import product
from pulp import LpVariable, lpSum, value
from ortoolpy import model_min, addvars, addvals

# 데이터 불러오기
df_tc = pd.read_csv('trans_cost.csv', index_col="공장")
df_demand = pd.read_csv('demand.csv')
df_supply = pd.read_csv('supply.csv')

# 초기 설정  #
np.random.seed(1)
nw = len(df_tc.index)
nf = len(df_tc.columns)
pr = list(product(range(nw), range(nf)))
```
- pulp: 최적화 모델 작성
- ortooply: 목적함수를 생성하여 최적화 문제를 품.

#### # 수리 모델 작성 
```python
m1 = model_min() # m1: 최소화를 실행하는 모델

# 목적함수를 정의
v1 = {(i,j):LpVariable('v%d_%d'%(i,j),lowBound=0) for i,j in pr}
m1 += lpSum(df_tc.iloc[i][j]*v1[i,j] for i,j in pr)  

# 제약 조건을 정의
for i in range(nw):
    m1 += lpSum(v1[i,j] for j in range(nf)) <= df_supply.iloc[0][i]
for j in range(nf):
    m1 += lpSum(v1[i,j] for i in range(nw)) >= df_demand.iloc[0][j]
m1.solve() # 변수 v1이 최적화됨.
```

> - LpVariable: dict 형식으로 정의
> - 1pSum:   
- 목적함수: 각 운송 경로의 비용을 저장한 df_tc와 주요 변수 v1의 각 요소의 곱의 합으로 정의 
- 제약조건
    - 공장 - 제조할 제품 수요량을 만족시킴.
    - 창고 - 제공할 부품이 제공 한계를 넘지 않도록 함. 

#### # 총 운송 비용 계산
```python
df_tr_sol = df_tc.copy()
total_cost = 0
for k,x in v1.items():
    i,j = k[0],k[1]
    df_tr_sol.iloc[i][j] = value(x)
    total_cost += df_tc.iloc[i][j]*value(x)
    
print(df_tr_sol)
print("총 운송 비용:"+str(total_cost))
```


### **62. 최적 운송 경로를 네트워크로 확인하자.**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# 데이터 불러오기
df_tr = df_tr_sol.copy()
df_pos = pd.read_csv('trans_route_pos.csv')

# 객체 생성
G = nx.Graph()

# 노드 설정
for i in range(len(df_pos.columns)):
    G.add_node(df_pos.columns[i])

# 엣지 설정 & 엣지의 가중치 리스트화
num_pre = 0
edge_weights = []
size = 0.1
for i in range(len(df_pos.columns)):
    for j in range(len(df_pos.columns)):
        if not (i==j):
            # 엣지 추가
            G.add_edge(df_pos.columns[i],df_pos.columns[j])
            # 엣지 가중치 추가
            if num_pre<len(G.edges):
                num_pre = len(G.edges)
                weight = 0
                if (df_pos.columns[i] in df_tr.columns)and(df_pos.columns[j] in df_tr.index):
                    if df_tr[df_pos.columns[i]][df_pos.columns[j]]:
                        weight = df_tr[df_pos.columns[i]][df_pos.columns[j]]*size
                elif(df_pos.columns[j] in df_tr.columns)and(df_pos.columns[i] in df_tr.index):
                    if df_tr[df_pos.columns[j]][df_pos.columns[i]]:
                        weight = df_tr[df_pos.columns[j]][df_pos.columns[i]]*size
                edge_weights.append(weight)
                

# 좌표 설정
pos = {}
for i in range(len(df_pos.columns)):
    node = df_pos.columns[i]
    pos[node] = (df_pos[node][0],df_pos[node][1])
    
# 그리기
nx.draw(G, pos, with_labels=True,font_size=16, node_size = 1000, node_color='k', font_color='w', width=edge_weights)

# 표시
plt.show()
```
![network_draw](/assets/img/network_draw.jpg)   
-> 일부 경로에 집중되어 있음.

### **63. 최적 운송 경로가 제약 조건을 만족하는지 확인하자.**

```python
import pandas as pd
import numpy as np

# 데이터 불러오기
df_demand = pd.read_csv('demand.csv')
df_supply = pd.read_csv('supply.csv')
```

#### # 제약조건 계산함수
```python
# 수요측
def condition_demand(df_tr,df_demand):
    flag = np.zeros(len(df_demand.columns))
    for i in range(len(df_demand.columns)):
        temp_sum = sum(df_tr[df_demand.columns[i]])
        if (temp_sum>=df_demand.iloc[0][i]):
            flag[i] = 1
    return flag
            
# 공급측
def condition_supply(df_tr,df_supply):
    flag = np.zeros(len(df_supply.columns))
    for i in range(len(df_supply.columns)):
        temp_sum = sum(df_tr.loc[df_supply.columns[i]])
        if temp_sum<=df_supply.iloc[0][i]:
            flag[i] = 1
    return flag

print("수요 조건 계산 결과:"+str(condition_demand(df_tr_sol,df_demand))) # 출력: [1. 1. 1. 1.]
print("공급 조건 계산 결과:"+str(condition_supply(df_tr_sol,df_supply))) # 출력: [1. 1. 1.]
```
-> 모든 제약 조건이 충족됨.

### **64. 생산 계획 데이터를 불러오자.**

```python
import pandas as pd

df_material = pd.read_csv('product_plan_material.csv', index_col="제품")
print(df_material)
df_profit = pd.read_csv('product_plan_profit.csv', index_col="제품")
print(df_profit)
df_stock = pd.read_csv('product_plan_stock.csv', index_col="항목")
print(df_stock)
df_plan = pd.read_csv('product_plan.csv', index_col="제품")
print(df_plan)
```
![before_produce](/assets/img/before_produce.jpg)    
-> 이익이 큰 제품1만 생산중   
-> 원료가 효과적으로 사용되지 않고 있기 때문에 제품 2의 생산량을 늘린다면 이익을 높일 수 있을 것

### **65. 이익을 계산하는 함수를 만들자.**
- product_plan_material.csv: 제품 제조에 필요한 원료 비율
- product_plan_profit.csv: 제품 이익
- product_plan_stock.csv: 원료 재고
- product_plan.csv: 제품 생산량

```python
# 이익 계산 함수
def product_plan(df_profit,df_plan):
    profit = 0
    for i in range(len(df_profit.index)):
        for j in range(len(df_plan.columns)):
            profit += df_profit.iloc[i][j]*df_plan.iloc[i][j]
    return profit

print("총 이익:"+str(product_plan(df_profit,df_plan)))
```

### **66. 생산 최적화 문제를 풀어보자.**
- 생산계획의 총 이익 = (각 제품의 이익 X 제조량) 의 합

```python
import pandas as pd
from pulp import LpVariable, lpSum, value
from ortoolpy import model_max, addvars, addvals


df = df_material.copy()
inv = df_stock

m = model_max() # 최대화 계산 준비 

v1 = {(i):LpVariable('v%d'%(i),lowBound=0) for i in range(len(df_profit))}
m += lpSum(df_profit.iloc[i]*v1[i] for i in range(len(df_profit)))
for i in range(len(df_material.columns)):
    m += lpSum(df_material.iloc[j,i]*v1[j] for j in range(len(df_profit)) ) <= df_stock.iloc[:,i]
m.solve()

df_plan_sol = df_plan.copy()
for k,x in v1.items():
    df_plan_sol.iloc[k] = value(x)
print(df_plan_sol)
print("총 이익:"+str(value(m.objective)))
```
- 변수 v1: 제품 수
- 목적함수: 변수 v1과 제품별 이익의 곱의 합
- 제약조건: 각 원료의 사용량이 재고를 넘지 않도록 함.   
-> 제품 1의 생산량을 1 줄이고, 제품 2의 생산량을 5 늘림.   
-> 이익 95만원으로 증가


### **67. 최적 생산 계획이 제약 조건을 만족하는지 확인하자.**
목적함수와 제약 조건이 현실과 달라 계산된 결과가 현실에 안 맞는 경우를 피하기 위해   
-> 제약 조건으로 규정한 '각 원료의 사용량'이 어느 정도인지 알아보기   
-> '재고를 효율적으로 이용하고 있는지' 알아보기

```python
# 제약 조건 계산 함수
def condition_stock(df_plan,df_material,df_stock):
    flag = np.zeros(len(df_material.columns))
    for i in range(len(df_material.columns)):  
        temp_sum = 0
        for j in range(len(df_material.index)):  
            temp_sum = temp_sum + df_material.iloc[j][i]*float(df_plan.iloc[j])
        if (temp_sum<=float(df_stock.iloc[0][i])):
            flag[i] = 1
        print(df_material.columns[i]+"  사용량:"+str(temp_sum)+", 재고:"+str(float(df_stock.iloc[0][i])))
    return flag

print("제약 조건 계산 결과:"+str(condition_stock(df_plan_sol,df_material,df_stock)))
```
![constraint_check](/assets/img/constraint_check.jpg)   
-> 제약 조건 모두 충족   
-> 원료2, 원료3은 재고 모두 사용 / 원료1은 조금 남아 있음.   
=> 최적화 계산 전에 비해 원료의 사용 효율이 크게 개선됨. => 합리적

### **68. 물류 네트워크 설계 문제를 풀어보자.**
- 물류 네트워크는 운송 경로와 생산 계획 최적화 문제를 동시에 고려해야 함.
- 상품의 수요가 정해져 있다면, 비용을 최소화 하는 것이 중요.   
=> 운송 비용과 제조 비용이 수요를 만족하면서 최소가 되도록 정식화
    - 목적함수: 운송 비용 + 제조 비용
    - 제약조건: 각 대리점의 판매 수가 수요 수를 넘음.

```python
import numpy as np
import pandas as pd

제품 = list('AB')
대리점 = list('PQ')
공장 = list('XY')
레인 = (2,2)

# 운송비 #
tbdi = pd.DataFrame(((j,k) for j in 대리점 for k in 공장), columns=['대리점','공장'])
tbdi['운송비'] = [1,2,3,1]
print(tbdi)

# 수요 #
tbde = pd.DataFrame(((j,i) for j in 대리점 for i in 제품), columns=['대리점','제품'])
tbde['수요'] = [10,10,20,20]
print(tbde)

# 생산 #
tbfa = pd.DataFrame(((k,l,i,0,np.inf) for k,nl in zip (공장,레인) for l in range(nl) for i in 제품), 
                    columns=['공장','레인','제품','하한','상한'])
tbfa['생산비'] = [1,np.nan,np.nan,1,3,np.nan,5,3]
tbfa.dropna(inplace=True)
tbfa.loc[4,'상한']=10
print(tbfa)

from ortoolpy import logistics_network
_, tbdi2, _ = logistics_network(tbde, tbdi, tbfa,dep = "대리점", dem = "수요",fac = "공장",
                                prd = "제품",tcs = "운송비",pcs = "생산비",lwb = "하한",upb = "상한")

print(tbfa)
print(tbdi2)
```
> logistics_network
>   - 생산표에 ValY라는 항목이 만들어지면서 최적 생산량 저장.
>   - 운송 비표에 ValX라는 항목이 만들어지고 최적 운송량 저장.

### **69. 최적 네트워크의 운송 비용과 그 내역을 계산하자.**


```python
tbdi2 = tbdi2[["공장","대리점","운송비","제품","VarX","ValX"]]
tbdi2
```
- 운송 비용은 함수 logistics_network의 반환값으로 tbdi2에 저장됨.


```python
trans_cost = 0
for i in range(len(tbdi2.index)):
    trans_cost += tbdi2["운송비"].iloc[i]*tbdi2["ValX"].iloc[i]
print("총 운송비:"+str(trans_cost)) # 80만원
```
- 운송 비용 = 운송비 X 최적 운송량(ValX)   
-> 되도록 운송비가 적은 경로를 사용하고, 수요량이 많은 대리점의 수요를 고려하여 다른 경로를 추가적으로 이용.


### **70. 최적 네트워크의 생산 비용과 그 내역을 계산하자.**

```python
tbfa
```
- 생산 비용은 logistics_network 함수 계산 후 tbfa에 저장됨.

```python
product_cost = 0
for i in range(len(tbfa.index)):
    product_cost += tbfa["생산비"].iloc[i]*tbfa["ValY"].iloc[i]
print("총 생산비:"+str(product_cost)) # 120만원
```
-> 생산 비용이 낮은 공장에서의 생산량을 늘림.   
-> 운송 비용까지 고려하여 수요량이 많은 대리점까지의 운송 비용이 적은 공장의 생산량 결정.