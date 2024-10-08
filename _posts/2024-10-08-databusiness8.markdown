---
layout: post
title: "[파이썬 데이터분석 실무 테크닉 100] 08장. 수치 시뮬레이션으로 소비자의 행동을 예측하는 테크닉 10
date: 2023-10-08
categories: "데이터비지니스"
---

- links.csv: 재구매 고객 20명의 SNS 연결 상태 (연결 1, 연결X 0)
- links_members.csv: 재구매 고객 540명의 SNS 연결 상태
- info_members.csv: 재구매 고객 540명의 월별 이용 현황 (이용 실적 있는 달 1)


### **71. 인간관계 네트워크를 가시화해 보자.**

```python
import pandas as pd

df_links = pd.read_csv("links.csv")
df_links
```
#### # 가시화
```python
import networkx as nx
import matplotlib.pyplot as plt

# 그래프 객체 생성
G = nx.Graph()

# 노드 설정
NUM = len(df_links.index)
for i in range(1,NUM+1):
    node_no = df_links.columns[i].strip("Node")
    G.add_node(str(node_no))

# 엣지 설정
for i in range(NUM):
    for j in range(NUM):
        node_name = "Node" + str(j)
        if df_links[node_name].iloc[i]==1:
            G.add_edge(str(i),str(j))
        
# 그리기
nx.draw_networkx(G,node_color="k", edge_color="k", font_color="w")
plt.show()
```
> draw_networkx: 다른 것과 연결이 많은 노드를 중심에 오게 자동으로 위치 결정


### **72. 입소문에 의한 정보 전파 과정의 모습을 가시화해 보자.**
가정: 10개의 연결 중 하나의 확률 (10%의 확률)로 소문이 전파된다.

```python
import numpy as np
```
```python
def determine_link(percent):
    rand_val = np.random.rand()
    if rand_val<=percent:
        return 1
    else:
        return 0
```
-> 입소문의 전파 여부를 확률적으로 결정   
-> 인수로 입소문을 낼 확률 전달

```python
def simulate_percolation(num, list_active, percent_percolation):
    for i in range(num):
        if list_active[i]==1:
            for j in range(num):
                node_name = "Node" + str(j)
                if df_links[node_name].iloc[i]==1:
                    if determine_link(percent_percolation)==1:
                        list_active[j] = 1
    return list_active
```
-> 입소문을 시뮬레이션함.   
--> num: 사람 수   
--> list_active: 각각의 노드에 입소문이 전달됐는지 (1 또는 0)   
--> percent_percolation: 입소문을 일으킬 확률

```python
percent_percolation = 0.1
T_NUM = 36 # 36개월 반복
NUM = len(df_links.index)
list_active = np.zeros(NUM)
list_active[0] = 1

list_timeSeries = []
for t in range(T_NUM):
    list_active = simulate_percolation(NUM, list_active, percent_percolation)
    list_timeSeries.append(list_active.copy())
```

#### # 액티브 노드 가시화
```python
def active_node_coloring(list_active):
    #print(list_timeSeries[t])
    list_color = []
    for i in range(len(list_timeSeries[t])):
        if list_timeSeries[t][i]==1:
            list_color.append("r")
        else:
            list_color.append("k")
    #print(len(list_color))
    return list_color
```
- active_node_coloring: 입소문이 전파된(활성화된) 노드는 빨간색으로, 전파되지 않은 노드는 검은색으로 색칠

```python
# 그리기
t = 0
nx.draw_networkx(G,font_color="w",node_color=active_node_coloring(list_timeSeries[t]))
plt.show()
```
![active_node_0](/assets/img/active_node_0.jpg)

```python
# 그리기
t = 11
nx.draw_networkx(G,font_color="w",node_color=active_node_coloring(list_timeSeries[t]))
plt.show()
```
![active_node_11](/assets/img/active_node_11.jpg)

```python
# 그리기
t = 35
nx.draw_networkx(G,font_color="w",node_color=active_node_coloring(list_timeSeries[t]))
plt.show()
```
![active_node_35](/assets/img/active_node_35.jpg)


### **73. 입소문 수의 시계열 변화를 그래프화해 보자.**

```python
# 시계열 그래프 그리기
list_timeSeries_num = []
for i in range(len(list_timeSeries)):
    list_timeSeries_num.append(sum(list_timeSeries[i]))

plt.plot(list_timeSeries_num)
plt.show()
```
![timeseries_num](/assets/img/timeseries_num.jpg)

### **74. 회원 수의 시계열 변화를 시뮬레이션해 보자.**

```python
def simulate_population(num, list_active, percent_percolation, percent_disapparence,df_links):
    # 확산 #
    for i in range(num):
        if list_active[i]==1:
            for j in range(num):
                if df_links.iloc[i][j]==1:
                    if determine_link(percent_percolation)==1:
                        list_active[j] = 1
    # 소멸 #
    for i in range(num):
        if determine_link(percent_disapparence)==1:
            list_active[i] = 0
    return list_active
```
- simulate_population
    - '소멸'이라는 조작 추가
    - 기존의 회원이 5%의 확률로 갑자기 탈퇴한다고 가정.

#### # 소멸 확률 5%
```python
percent_percolation = 0.1
percent_disapparence = 0.05
T_NUM = 100
NUM = len(df_links.index)
list_active = np.zeros(NUM)
list_active[0] = 1

list_timeSeries = []
for t in range(T_NUM):
    list_active = simulate_population(NUM, list_active, percent_percolation, percent_disapparence,df_links)
    list_timeSeries.append(list_active.copy())
```

```python
# 시계열 그래프 그리기
list_timeSeries_num = []
for i in range(len(list_timeSeries)):
    list_timeSeries_num.append(sum(list_timeSeries[i]))

plt.plot(list_timeSeries_num)
plt.show()
```
![disapparence5](/assets/img/disapparence5.jpg)   
-> 증감을 반복하며 100% 이용률을 향해감.

#### # 소멸 확률 20%
```python
percent_disapparence = 0.2
list_active = np.zeros(NUM)
list_active[0] = 1
list_timeSeries = []
for t in range(T_NUM):
    list_active = simulate_population(NUM, list_active, percent_percolation, percent_disapparence,df_links)
    list_timeSeries.append(list_active.copy())
```

```python
# 시계열 그래프 그리기
list_timeSeries_num = []
for i in range(len(list_timeSeries)):
    list_timeSeries_num.append(sum(list_timeSeries[i]))

plt.plot(list_timeSeries_num)
plt.show()
```
![disapparence20](/assets/img/disapparence20.jpg)   
-> 40개월 후에는 이용자가 없어짐.




### **75. 파라미터 전체를 '상관관계'를 보면서 파악해 보자.**

확산과 소멸이 일어나는 확률이 상품 보급에 어떻게 영향을 주는지 '상관관계'를 통해 파악.

#### # 상관관계 계산
```python
print("상관관계 계산시작")
T_NUM = 100
NUM_PhaseDiagram = 20
phaseDiagram = np.zeros((NUM_PhaseDiagram,NUM_PhaseDiagram))
for i_p in range(NUM_PhaseDiagram):
    for i_d in range(NUM_PhaseDiagram):
        percent_percolation = 0.05*i_p
        percent_disapparence = 0.05*i_d
        list_active = np.zeros(NUM)
        list_active[0] = 1
        for t in range(T_NUM):
            list_active = simulate_population(NUM, list_active, percent_percolation, percent_disapparence,df_links)
        phaseDiagram[i_p][i_d] = sum(list_active)
print(phaseDiagram)
```
#### # 표시
```python
plt.matshow(phaseDiagram)
plt.colorbar(shrink=0.8)
plt.xlabel('percent_disapparence')
plt.ylabel('percent_percolation')
plt.xticks(np.arange(0.0, 20.0,5), np.arange(0.0, 1.0, 0.25))
plt.yticks(np.arange(0.0, 20.0,5), np.arange(0.0, 1.0, 0.25))
plt.tick_params(bottom=False,
                left=False,
                right=False,
                top=False)
plt.show()
```
![phaseDiagram](/assets/img/phaseDiagram.jpg)   
-> 확산,소멸 확률을 조금씩 변화시키며 100개월 후 이용자 수를 색으로 표시   
--> 소멸 확률이 낮을 때 - 확산 확률이 낮아도 20명 전원이 이용 중   
--> 소멸 확률이 20~30% 이상 - 확산 확률이 높아도 이용자 증가 X

### **76. 실제 데이터를 불러와보자.**

- links_members.csv: 540명의 SNS 연결 저장
- info_members.csv: 540auddml 24개월간 이용 현황 (이용 내역 있는 달 1)
```python
import pandas as pd

df_mem_links = pd.read_csv("links_members.csv")
df_mem_info = pd.read_csv("info_members.csv")
df_mem_links
```

### **77. 링크 수의 분포를 가시화해 보자.**

```python
NUM = len(df_mem_links.index)
array_linkNum = np.zeros(NUM)
for i in range(NUM):
    array_linkNum[i] = sum(df_mem_links["Node"+str(i)])
```

```python
plt.hist(array_linkNum, bins=10,range=(0,250))
plt.show()
```
![link_hist](/assets/img/link_hist.jpg)

- hist 함수로 링크 개수의 히스토그램 표시   
-> 링크 개수가 대략 100 정도에 집중된 정규분포에 가까움.

- 거의 모든 노드가 어느 정도의 링크 수를 가지고 있는 분포   
-> 허브에 의존하지 않고 입소문이 퍼지기 쉬움.

- if 스케일 프리형이라면?   
-> 링크를 많이 가진 허브가 작동하지 않으면 확산되지 않음.

>- 스몰 월드형: 몇 안 되는 스텝으로 전원이 연결
>- 스케일 프리형: 소수의 연결을 많이 가지는 사람이 허브가 됨.

### **78. 시뮬레이션을 위해 실제 데이터로부터 파라미터를 추정하자.**

시뮬레이션을 하기 위해서는 먼저 데이터를 이용해서 파라미터를 추정해야 함.

```python
NUM = len(df_mem_info.index)
T_NUM = len(df_mem_info.columns)-1
```

#### # 소멸 확률 추정
```python
count_active = 0
count_active_to_inactive = 0
for t in range(1,T_NUM):
    for i in range(NUM):
        if (df_mem_info.iloc[i][t]==1):
            count_active_to_inactive += 1
            if (df_mem_info.iloc[i][t+1]==0):
                count_active += 1
estimated_percent_disapparence = count_active/count_active_to_inactive
```
-> df_mem_info를 시계열순으로 보고 1스텝 전과 비교하여 활성(1)이 비활성(0)으로 변화한 비율을 셈.

#### # 확산 확률 추정

```python
count_link = 0
count_link_to_active = 0
count_link_temp = 0
for t in range(T_NUM-1):
    df_link_t = df_mem_info[df_mem_info[str(t)]==1]
    temp_flag_count = np.zeros(NUM)
    for i in range(len(df_link_t.index)):
        df_link_temp = df_mem_links[df_mem_links["Node"+str(df_link_t.index[i])]==1]
        for j in range(len(df_link_temp.index)):
            if (df_mem_info.iloc[df_link_temp.index[j]][t]==0):
                if (temp_flag_count[df_link_temp.index[j]]==0):
                    count_link += 1
                if (df_mem_info.iloc[df_link_temp.index[j]][t+1]==1):
                    if (temp_flag_count[df_link_temp.index[j]]==0):
                        temp_flag_count[df_link_temp.index[j]] = 1 
                        count_link_to_active += 1
estimated_percent_percolation = count_link_to_active/count_link
```
-> 어떤 노드가 비활성에서 활성 상태로 변하는 것은 링크 개수와 관계없이 발생.  
=> 비활성이나 활성 개수를 세고 그 비율로부터 확률을 추정하는 방법 정확하지 X    
=> 중복해서 세지 않도록 함.

```python
estimated_percent_disapparence # 출력: 0.10147...
```
```python
estimated_percent_percolation # 출력: 0.02518...
```


### **79. 실제 데이터와 시뮬레이션을 비교하자.**

```python
percent_percolation = 0.025184661323275185
percent_disapparence = 0.10147163541419416
T_NUM = 24
NUM = len(df_mem_links.index)
list_active = np.zeros(NUM)
list_active[0] = 1
list_timeSeries = []
for t in range(T_NUM):
    list_active = simulate_population(NUM, list_active, percent_percolation, percent_disapparence,df_mem_links)
    list_timeSeries.append(list_active.copy())
```

```python
list_timeSeries_num = []
for i in range(len(list_timeSeries)):
    list_timeSeries_num.append(sum(list_timeSeries[i]))
```

```python
T_NUM = len(df_mem_info.columns)-1
list_timeSeries_num_real = []
for t in range(0,T_NUM):
    list_timeSeries_num_real.append(len(df_mem_info[df_mem_info[str(t)]==1].index))
```

```python
plt.plot(list_timeSeries_num, label = 'simulated')
plt.plot(list_timeSeries_num_real, label = 'real')
plt.xlabel('month')
plt.ylabel('population')
plt.legend(loc='lower right')
plt.show()
```
![simulated_vs_real](/assets/img/simulated_vs_real.jpg)   
-> 시뮬레이션으로 어느 지점에서 이용자가 급격히 증가하는지의 경향은 확인되지만 프로그램 난수의 영향으로 증가하는 시간대나 시기에 오차가 생김.   
=> 같은 시뮬레이션을 여러 번 실행해서 평균값을 계산하는 것이 좋음.

### **80. 시뮬레이션으로 미래를 예측해 보자.**


```python
percent_percolation = 0.025184661323275185
percent_disapparence = 0.10147163541419416
T_NUM = 36 # 시뮬레이터의 지속시간 36개월
NUM = len(df_mem_links.index)
list_active = np.zeros(NUM)
list_active[0] = 1
list_timeSeries = []
for t in range(T_NUM):
    list_active = simulate_population(NUM, list_active, percent_percolation, percent_disapparence,df_mem_links)
    list_timeSeries.append(list_active.copy())
```

```python
list_timeSeries_num = []
for i in range(len(list_timeSeries)):
    list_timeSeries_num.append(sum(list_timeSeries[i]))
```

```python
plt.plot(list_timeSeries_num, label = 'simulated')
plt.xlabel('month')
plt.ylabel('population')
plt.legend(loc='lower right')
plt.show()
```
![simulated](/assets/img/simulated.jpg)   
-> 24개월 이후에는 population이 급격히 감소하지 않고 지속될 것이다. 