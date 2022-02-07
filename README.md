# ARP-spoofing-detection
(2020 3-1) 사이버물리시스템보안 - IoT 침입탐지시스템 개발 과제

## 설계 목적
* 네트워크 패킷 dataset을 활용하여 ARP spoofing에 대한 딥러닝 기반 IoT 침입탐지 시스템 구현

## 트레이닝 데이터셋
### 데이터셋 정보
* IoT 기기의 네트워크 패킷을 캡쳐한 dataset을 활용 (csv파일 6개)
* ARP spoofing 공격이 발생한 경우와 정상 운영된 경우의 네트워크 패킷을 모두 포함

|FileNmae|Creation Date|Target Device|#Total Packets|# Attack Packets|
|------|:---:|:---:|---:|---:|
|mitm-arpspoofing-1-dec.pcap|5/31/2019|EZVIZ|65,768|34,855| 
|mitm-arpspoofing-2-dec.pcap|5/31/2019|EZVIZ|33,121|13,134|
|mitm-arpspoofing-3-dec.pcap|5/31/2019|EZVIZ|34,043|15,144|
|mitm-arpspoofing-4-dec.pcap|6/3/2019|NUGU|19,914|13,211|
|mitm-arpspoofing-5-dec.pcap|6/3/2019|NUGU|20,314|9,743|
|mitm-arpspoofing-6-dec.pcap|6/3/2019|NUGU|21,024|15,798|

※데이터셋 출처 : http://ocslab.hksecurity.net/Datasets/iot-network-intrusion-dataset
<br>

### 데이터 선별
* ARP spoofing은 mac관련 공격이며, 다른 column들은 결측치가 많기 때문에 제공된 패킷 데이터의 20가지 칼럼 중 다음 7개 칼럼을 학습에 사용
> ip.src, ip.dst, eth.src, eth.dst, arp.src.hw_mac, arp.dst.hw_mac, label  
<img src="https://user-images.githubusercontent.com/58112670/152847022-0e6be8c0-b812-4ec2-be23-7fefa1fe7a69.png" width="850">

## 학습 알고리즘 설계
### 1. ExtraTressClassfier 모델
```
model = ExtraTreeClassifier(n_estimators=100, criterion='entropy', random_state=2020)
```

### 2. SimpleRNN 모델
```
model = Sequential()
model.add(Embedding(400,64))    #임베딩 벡터의 차원은 64
model.add(Dense(64, activation='relu'))
model.add(SimpleRNN(32))      #LSTM 셀의 hidden_size는 32
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
```

## 성능
7회의 cross validation으로 Acurracy, Precision, Recall Score, F1-Score 평균 결과 분석
### 1. ExtraTressClassfier 모델
<img src="https://user-images.githubusercontent.com/58112670/152842631-e8ff22a2-e153-4e1c-9eb0-33cb399811f4.png" width="200"/>  

### 2. SimpleRNN 모델
<img src="https://user-images.githubusercontent.com/58112670/152842614-8fe65811-60d6-4ca6-818f-7b9b71161023.png" width="200"/>

## 고찰
* 단순한 딥러닝 네트워크로도 높은 정확도의 ARP Spoofing 탐지 모델을 학습 시킬 수 있었음 
* 보안의 다양한 분야에 딥러닝 기술이 활용될 여지가 많음
* 결측치 처리 방법(제거/대체, 대체값 선정 etc..)에 대한 고민 필요
* 과적합 방지를 위해 earlystopping 등 다양한 최적화 방법 고민할 필요가 있음
