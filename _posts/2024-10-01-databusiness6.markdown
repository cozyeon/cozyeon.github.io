---
layout: post
title: "[파이썬 데이터분석 실무 테크닉 100] 06장. 물류의 최적경로를 컨설팅하는 테크닉 10"
date: 2023-10-01
categories: "데이터비지니스"
---

- tbl.factory.csv: 생산 공장 데이터
- tbl_warehouse.csv: 창고 데이터
- rel_cost.csv: 창고와 공장 간의 운송 비용
- tbl_transaction.csv: 2019년의 공장으로의 부품 운송 실적

### **51. 물류 데이터를 불러오자.**

```python
import pandas as pd

# 공장데이터 불러오기
factories = pd.read_csv("tbl_factory.csv", index_col=0)
factories

# 창고데이터 불러오기
warehouses = pd.read_csv("tbl_warehouse.csv", index_col=0)
warehouses

# 비용 테이블
cost = pd.read_csv("rel_cost.csv", index_col=0)
cost.head()


# 운송 실적 테이블
trans = pd.read_csv("tbl_transaction.csv", index_col=0)
trans.head()
```

#### # 운송실적 테이블에 각 테이블을 조인
```python
# 비용 데이터추가
join_data = pd.merge(trans, cost, left_on=["ToFC","FromWH"], right_on=["FCID","WHID"], how="left")
join_data.head()

# 공장정보 추가
join_data = pd.merge(join_data, factories, left_on="ToFC", right_on="FCID", how="left")
join_data.head()

# 창고정보 추가
join_data = pd.merge(join_data, warehouses, left_on="FromWH", right_on="WHID", how="left")

# 컬럼 순서 정리
join_data = join_data[["TransactionDate","Quantity","Cost","ToFC","FCName","FCDemand","FromWH","WHName","WHSupply","WHRegion"]]
join_data.head()

# 북부 데이터 추출
north = join_data.loc[join_data["WHRegion"]=="북부"]
north.head()

# 남부데이터 추출
south = join_data.loc[join_data["WHRegion"]=="남부"]
south.head()
```

### **52. 현재 운송량과 비용을 확인해 보자.**

```python
# 지사의 비용합계 계산
print("북부지사 총비용: " + str(north["Cost"].sum()) + "만원")
print("남부지사 총비용: " + str(south["Cost"].sum()) + "만원")

# 지사의 총운송개수
print("북부지사의 총부품 운송개수: " + str(north["Quantity"].sum()) + "개")
print("남부지사의 총부품 운송개수: " + str(south["Quantity"].sum()) + "개")

# 부품 1개당 운송비용
tmp = (north["Cost"].sum() / north["Quantity"].sum()) * 10000
print("북부지사의 부품 1개당 운송 비용: " + str(int(tmp)) + "원")  # 북부지사의 부품 1개당 운송비용: 445원
tmp = (south["Cost"].sum() / south["Quantity"].sum()) * 10000
print("남부지사의 부품 1개당 운송 비용: " + str(int(tmp)) + "원")  # 남부지사의 부품 1개당 운송비용: 410원

# 비용을 지사별로 집계
cost_chk = pd.merge(cost, factories, on="FCID", how="left") 

# 평균
print("북부지사의 평균 운송 비용：" + str(cost_chk["Cost"].loc[cost_chk["FCRegion"]=="북부"].mean()) + "원") # 북부지사의 평균 운송 비용 : 1.075원
print("남부지사의 평균 운송 비용：" + str(cost_chk["Cost"].loc[cost_chk["FCRegion"]=="남부"].mean()) + "원") # 남부지사의 평균 운송 비용 : 1.05원
```
-> 북부지사보다 남부지사가 '효율 높게' 부품을 운송하고 있음.

### **53. 네트워크를 가시화해 보자.**
- 엣지: 노드끼리 연결하는 선
```python
import networkx as nx
import matplotlib.pyplot as plt

# 그래프 객체생성
G=nx.Graph()

# 노드 설정
G.add_node("nodeA")
G.add_node("nodeB")
G.add_node("nodeC")

# 엣지 설정
G.add_edge("nodeA","nodeB")
G.add_edge("nodeA","nodeC")
G.add_edge("nodeB","nodeC")

# 좌표 설정
pos={}
pos["nodeA"]=(0,0)
pos["nodeB"]=(1,1)
pos["nodeC"]=(0,1)

# 그리기
nx.draw(G,pos)

# 표시
plt.show()
```
![network1](/assets/img/network1.jpg)

### **54. 네트워크에 노드를 추가해 보자.**

```python
import networkx as nx
import matplotlib.pyplot as plt

# 그래프 객체 생성．
G=nx.Graph()

# 노드 설정
G.add_node("nodeA")
G.add_node("nodeB")
G.add_node("nodeC")
G.add_node("nodeD")

# 엣지 설정
G.add_edge("nodeA","nodeB")
G.add_edge("nodeA","nodeC")
G.add_edge("nodeB","nodeC")
G.add_edge("nodeA","nodeD")

# 좌표 설정
pos={}
pos["nodeA"]=(0,0)
pos["nodeB"]=(1,1)
pos["nodeC"]=(0,1)
pos["nodeD"]=(1,0)

# 그리기
nx.draw(G,pos, with_labels=True)

# 표시
plt.show()
```
![network_node](/assets/img/network_node.jpg)

### **55. 경로에 가중치를 부여하자.**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# 데이터 불러오기
df_w = pd.read_csv('network_weight.csv')
df_p = pd.read_csv('network_pos.csv')

# 그래프 객체 생성
G = nx.Graph()

# 노드 설정
for i in range(len(df_w.columns)):
    G.add_node(df_w.columns[i])

# 엣지 설정 & 가중치 리스트화
size = 10
edge_weights = []
num_pre = 0

# 엣지 가중치 확인용 번역자 추가 코드
name = ['A','B','C','D','E']

for i in range(len(df_w.columns)):
    for j in range(len(df_w.columns)):
        if not (i==j):
            # 엣지 추가
            G.add_edge(df_w.columns[i],df_w.columns[j])
            if num_pre<len(G.edges):
                num_pre = len(G.edges)
                # 엣지 가중치 추가
                edge_weights.append(df_w.iloc[i][j]*size)
               
                # 엣지 가중치 확인용 번역자 추가 코드
                print(f'({name[i]}, {name[j]}) = {np.round(edge_weights[-1],5)}')
               

# 좌표 설정
pos = {}
for i in range(len(df_w.columns)):
    node = df_w.columns[i]
    pos[node] = (df_p[node][0],df_p[node][1])

# 그리기
nx.draw(G, pos, with_labels=True,font_size=16, node_size = 1000, node_color='k', font_color='w', width=edge_weights)

# 표시
plt.show()
```
![network_weight](/assets/img/network_weight.jpg)

### **56. 운송 경로 정보를 불러오자.**
- trans_route.csv: 운송 경로
- trans_route_pos.csv: 창고 및 공장의 위치 정보
- trans_cost.csv: 창고와 공장 간의 운송 비용
- demand.csv: 공장의 제품 생산량에 대한 수요
- supply.csv: 창고가 공급 가능한 최대 부품 수
- trans_route_new.csv: 새로 설계한 운송 경로 
- W1, W2, W3: 제품을 저장한 창고
- F1, F2, F3, F4: 부품을 운송해야 하는 조립 공장

```python
import pandas as pd

df_tr = pd.read_csv('trans_route.csv', index_col="공장")
df_tr.head()
```

### **57. 운송 경로 정보로 네트워크를 가시화해 보자.**

```python
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

df_tr = pd.read_csv('trans_route.csv', index_col="공장")
df_pos = pd.read_csv('trans_route_pos.csv')


# 그래프 객체 생성
G = nx.Graph()

# 노드 설정
for i in range(len(df_pos.columns)):
    G.add_node(df_pos.columns[i])

# 엣지 설정 및 가중치 리스트화
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
![network2](/assets/img/network2.jpg)


### **58. 운송 비용함수를 작성하자.**

>최적화 문제
>-
>- 목적함수: 최소화(최대화)하고 싶은 것을 함수로 정의 
>- 제약조건: 최소화(최대화)를 함에 있어 지켜야 할 조건을 정의   
>-> 여러 운송 경로의 조합 중 제약 조건을 만족시키면서 목적함수를 최소화(최대화)하는 조합을 선택

```python
import pandas as pd

# 데이터 불러오기
df_tr = pd.read_csv('trans_route.csv', index_col="공장")
df_tc = pd.read_csv('trans_cost.csv', index_col="공장")

# 운송 비용 함수
def trans_cost(df_tr,df_tc):
    cost = 0
    for i in range(len(df_tc.index)):
        for j in range(len(df_tr.columns)):
            cost += df_tr.iloc[i][j]*df_tc.iloc[i][j]
    return cost

print("총 운송 비용:"+str(trans_cost(df_tr,df_tc)))
```
-> 운송 비용 = 시그마 (운송량 x 비용)

### **59. 제약 조건을 만들어보자.**

- 창고는 공급 가능한 부품 수 제한
- 공장은 채워야 할 최소한의 제품 제조량

```python
import pandas as pd

# 데이터 불러오기
df_tr = pd.read_csv('trans_route.csv', index_col="공장")
df_demand = pd.read_csv('demand.csv')
df_supply = pd.read_csv('supply.csv')

# 수요측 제약조건
for i in range(len(df_demand.columns)):
    temp_sum = sum(df_tr[df_demand.columns[i]])
    print(str(df_demand.columns[i])+"으로 운송량:"+str(temp_sum)+" (수요량:"+str(df_demand.iloc[0][i])+")")
    if temp_sum>=df_demand.iloc[0][i]:
        print("수요량을 만족시키고있음")
    else:
        print("수요량을 만족시키지 못하고 있음. 운송경로 재계산 필요")

# 공급측 제약조건
for i in range(len(df_supply.columns)):
    temp_sum = sum(df_tr.loc[df_supply.columns[i]])
    print(str(df_supply.columns[i])+"부터의 운송량:"+str(temp_sum)+" (공급한계:"+str(df_supply.iloc[0][i])+")")
    if temp_sum<=df_supply.iloc[0][i]:
        print("공급한계 범위내")
    else:
        print("공급한계 초과. 운송경로 재계산 필요")
```

### **60. 운송 경로를 변경해서 운송 비용함수의 변화를 확인하자.**
W1 -> F4 운송 줄이고, W2 -> F4 운송 보충    
=> 제약 조건을 만족시키는지, 비용 개선 가능한지 계산

```python
import pandas as pd
import numpy as np

# 데이터 불러오기
df_tr_new = pd.read_csv('trans_route_new.csv', index_col="공장")
print(df_tr_new)

# 총 운송비용 재계산 
print("총 운송 비용(변경 후):"+str(trans_cost(df_tr_new,df_tc)))
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

print("수요조건 계산결과:"+str(condition_demand(df_tr_new,df_demand)))
print("공급조건 계산결과:"+str(condition_supply(df_tr_new,df_supply)))
```
-> 운송 비용 절감.   
-> W2의 공급 한계 넘음.