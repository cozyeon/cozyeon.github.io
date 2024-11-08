---
layout: post
title: "[통계101 데이터분석] Part 6,7"
date: 2023-10-01
categories: "데이터와통계학"
---

# 6장. 다양한 가설검정

## **6-1. 다양한 가설검정**

### 가설검정 방법 구분해 사용하기
#### # 데이터 유형
- 2개 변수 사이의 관계
![data_type](/assets/img/data_type.jpg)

#### # 표본의 수
![sample_number](/assets/img/sample_number.jpg)

#### # 양적 변수의 성질
- 모수검정: 모집단이 수학적으로 다룰 수 있는 특정 분포를 따른다는 가정을 둔 가설검정
- 정규성: 데이터가 정규분포로부터 얻어졌다고 간주할 수 있는 성질   
![normality](/assets/img/normality.jpg)
- 비모수검정: 파라미터에 기반을 두지 않는 검정
- 등분산성: 집단끼리 분산이 동일한 성질


## **6-2. 대푯값 비교**

### 정규성 조사
- Q-Q 플롯: 시각적으로 판단 가능
- 샤피로-윌크 검정: 가설검정으로 조사
- 콜모고로프-스미르노프(K-S) 검정: 이론적인 분포와 비교

### 등분산성 조사
- 바틀렛 검정
- 레빈 검정

### 비모수검정의 대푯값 비교
- 윌콕슨 순위합 검정, 맨-휘트니 U 검정: 평균값 대신 각 데이터 값의 순위에 기반하여 검정 실시   
-> 분포는 정규분포가 아니더라도 두 집단의 분포 모양 자체는 같아야 함.
- 플리그너-폴리셀로 검정, 브루너-문첼 검정   
-> 두 집단의 분포 형태가 같지 않아도 사용 가능

### 분산분석 (3개 집단 이상의 평균값 비교)
- F값 = (평균적인 집단 간 변동) / (평균적인 집단 낸 변동)   
-> 귀무가설이 올바르다는 가정 하에 F분포를 따름.
- p값: 이 분포에서 관찰한 F값 이상으로 극단적인 값이 나올 확률
![F_distribution](/assets/img/F_distribution.jpg)

> 자유도(degree of freedom): 자유로이 움직일 수 있는 변수의 수 

### 다중비교 검정
- 본페로니 교정: 전체에서 유의수준 a를 설정했을 때의 검정 반복 횟수 k, 매 검정에서 a/k를 기준으로 가설 검정   
-> p < a/k 일 때 대립가설 채택
- 튜키 검정: 본페로니 교정 보다 검정력 개선한 방법
- 던넷 검정: 대조군과의 비교에만 관심이 있을 때 이용
- 윌리엄스 검정: 집단 간에 순위를 매길 수 있는 경우에 이용
![multi_test](/assets/img/multi_test.jpg)


