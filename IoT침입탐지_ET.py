import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import precision_score,f1_score,recall_score

#arp 데이터셋을 불러오기
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
d1=pd.read_csv(r'C:/Users/clari/Desktop/ARP/DETECTION/CPS dataset/mitm-arpspoofing-1-dec.csv', encoding='utf-8')
d2=pd.read_csv(r'C:/Users/clari/Desktop/ARP/DETECTION/CPS dataset/mitm-arpspoofing-2-dec.csv', encoding='utf-8')
d3=pd.read_csv(r'C:/Users/clari/Desktop/ARP/DETECTION/CPS dataset/mitm-arpspoofing-3-dec.csv', encoding='utf-8')
d4=pd.read_csv(r'C:/Users/clari/Desktop/ARP/DETECTION/CPS dataset/mitm-arpspoofing-4-dec.csv', encoding='utf-8')
d5=pd.read_csv(r'C:/Users/clari/Desktop/ARP/DETECTION/CPS dataset/mitm-arpspoofing-5-dec.csv', encoding='utf-8')
d6=pd.read_csv(r'C:/Users/clari/Desktop/ARP/DETECTION/CPS dataset/mitm-arpspoofing-6-dec.csv', encoding='utf-8')

data=pd.concat([d1, d2, d3, d4, d5, d6])
frame_num=data['frame.number']

#학습에 사용할 데이터 선택
data=data.drop(['frame.number', 'icmp', 'ip.proto', 'tcp.srcport', 'tcp.dstport',
           'tcp.len', 'tcp.seq', 'tcp.ack', 'tcp.flags.push', 'tcp.flags.reset',
           'tcp.flags.syn', 'tcp.flags.fin', 'tcp.stream'], axis=1)

#결측치를 '0'으로 채우기
data = data.fillna('0')

#LabelEncoder) 정수 데이터로 인코딩
le=LabelEncoder()
data['ip.src']=le.fit_transform(data['ip.src'])
data['ip.dst']=le.fit_transform(data['ip.dst'])
data['eth.src']=le.fit_transform(data['eth.src'])
data['eth.dst']=le.fit_transform(data['eth.dst'])
data['arp.src.hw_mac']=le.fit_transform(data['arp.src.hw_mac'])
data['arp.dst.hw_mac']=le.fit_transform(data['arp.dst.hw_mac'])

#딥러닝 훈련 시 사용할 네트워크 구조
model = ExtraTreesClassifier(n_estimators=100,  criterion='entropy', random_state=2020)

#Cross validation
cv=KFold(n_splits=7,random_state=2020)
accs = [];  precision=[];  recall=[];  F1=[]
result=[]

for train_index,test_index in cv.split(data):
    #훈련용(6)
    x_train=data.iloc[train_index].drop(['label'], axis=1)
    y_train=data.iloc[train_index]['label']
    #검증용(1)
    x_test=data.iloc[test_index].drop(['label'], axis=1)
    y_test=data.iloc[test_index]['label']

    #훈련용에 대해 model 학습시키고(5)
    model.fit(x_train, y_train)

    #test로 model 검증(1)
    accs.append(model.score(x_test, y_test))
    precision.append(precision_score(y_test,model.predict(x_test).round()))
    recall.append(recall_score(y_test, model.predict(x_test).round()))
    F1.append(f1_score(y_test, model.predict(x_test).round()))

    # 각 폴드마다 탐지 결과를 csv파일에 저장
    tmp = model.predict(x_test)
    result.extend(tmp.flatten())

df = pd.DataFrame({'detection result': result, 'fame number': frame_num})
df.to_csv('TEST_RESULT_ET.csv', index=False)

#모델 정확도 평가
for i in range(1,8):
    print("Fold%d 훈련 결과 : "%i)
    print("     accs : %.6f"%accs[i-1])
    print("     precision : %.6f"%precision[i-1])
    print("     recall : %.6f"%recall[i-1])
    print("     F1 : %.6f"%F1[i-1])

print("\n평균 훈련 결과")
print("     Accuracy : %.6f"%np.mean(accs))
print("     Precision : %.6f"%np.mean(precision))
print("     Recall Score : %.6f"%np.mean(recall))
print("     F1-Score : %.6f"%np.mean(F1))