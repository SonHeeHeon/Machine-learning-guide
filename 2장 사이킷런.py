import sklearn
import pandas as pd
import numpy as np
print(sklearn.__version__)

# 2. 첫번째 모델 만들어보기

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris=load_iris()
iris_label=iris.target
print(iris.target_names)
print(iris.feature_names)
iris_data=iris.data
print(type(iris_data)) # 데이터를 넘파이의 ndarray 형태로 저장함
iris_df=pd.DataFrame(iris_data,columns=iris.feature_names)
iris_df.head(3)
iris_df['label']=iris.target
iris_df.head(3)
iris_df.info()

x_train,x_test,y_train,y_test=train_test_split(iris_df.iloc[:,:4],iris_df['label'],test_size=0.2,random_state=11) # 데이터 셋 분할 / X와 Y를 따로 기입해줘야함 / test_size 지정
dt_clf=DecisionTreeClassifier(random_state=11) # 머신러닝 객체 생성
dt_clf.fit(x_train,y_train) # Fit 함수로 학습시키기
pred=dt_clf.predict(x_test)
x=pred==y_test;print(x.sum()/len(x)) # 예측 정확도
print('예측 정확도 : ',accuracy_score(y_test,pred))

# 3. 사이킷런의 기반 프레임워크 익히기

## 사이킷런의 주요모듈
### 예제 데이터
import sklearn.datasets # 사이킷런에 내장되어 예제로 제공하는 데이터세트
### 피처 처리
import sklearn.preprocessing      # 데이터 전처리에 필요한 다양한 가공 기능 제공(ex- 문자열을 숫자형 코드값으로 인코딩 / 정규화 / 스케일링 등 )
import sklearn.feature_selection  # 알고리즘에 큰 영향을 미치는 피처를 우선순위대로 셀렉션 작업을 수행하는 다양한 기능 제공
import sklearn.feature_extraction # 텍스트 데이터나 이미지 데이터의 벡터화된 피처를 추출하는데 사용됨 / 예를 들어 count vectorizer 나 tf-idf vectorizer 등을 생성하는 기능 제공 / 텍스트는 sklearn.feature_extraction.text 이미지는 sklearn.feature_extraction.image
### 피처처리 & 차원축소
import sklearn.decomposition      # 차원 축소 관련 알고리즘을 지원하는 모듈 / PCA,NMF,Truncated SVD 등을 통해 차원축소 기능을 수행할 수 있음
### 데이터 분리,검증 & 파라미터 튜닝
import sklearn.model_selection    # 교차검증을 위한 학습용/테스트용 분리 , 그리드 서치 로 최적 파라미터 추출 제공
### 평가
import sklearn.metrics      # 분류,회귀,클러스터링,페어와이즈에 대한 다양한 성능 측정 방법 제공 / Accuracy,Precision,Recall,ROC-AUC,RMSE 등 제공
### ML 알고리즘
import sklearn.ensemble     # 앙상블 알고리즘 제공 / 랜포,에이다 부스트,그래디언 부스팅 등을 제공
import sklearn.linear_model # 주로 선형 회귀,릿지,라쏘 및 로지스틱 회귀 등 회귀 관련 알고리즘을 지원 , 또한 SGD 관련 알고리즘도 제공
import sklearn.naive_bayes  # 나이브 베이즈 알고리즘 제공, 가우시안 NB, 다항분포 NB 등.
import sklearn.neighbors    # 최근접 이웃 알고리즘 제공 , k-nn 등
import sklearn.svm          # 서포트 벡터 머신 알고리즘 제공
import sklearn.tree         # 의사 결정  트리 알고리즘 제공
import sklearn.cluster      # 비지도 클러스터링 알고리즘 제공 ( k-means / 계층형 / DBSCAN 등 )
### 유틸리티
import sklearn.pipeline     # 피처 처리 등의 변환과 ML 알고리즘 학습 , 예측 등을 함께 묶어서 실행 할 수 있는 유틸리티 제공

# 4. Model Selection 모듈 소개
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris=load_iris()
dt_clf=DecisionTreeClassifier()
train_data=iris.data
train_label=iris.target
dt_clf.fit(train_data,train_label)
pred=dt_clf.predict(train_data) # 훈련 데이터 검증하기
accuracy_score(train_label,pred)  # 훈련데이터 검증이기에 정확도 100프로

train_x,test_x,train_y,test_y=train_test_split(train_data,train_label,test_size=0.3,random_state=121,shuffle=True) # shuffle은 데이터를 분리하기 전에 미리 섞을지 => 디폴트는 true (데이터를 분산시켜 더 효율적인 학습)
dt_clf.fit(train_x,train_y)
pred=dt_clf.predict(test_x)
accuracy_score(test_y,pred)

## 교차검증
### k fold 교차검증
from sklearn.model_selection import KFold

iris=load_iris()
features=iris.data
label=iris.target
dt_clf=DecisionTreeClassifier(random_state=156)
kfold=KFold(n_splits=5,random_state=154)
cv_accuracy=[]
features.shape # 붓꽃 데이터 크기 : 150개 row
n_iter=0
for train_index,test_index in kfold.split(features):
    x_train,x_test=features[train_index],features[test_index]
    y_train,y_test=label[train_index],label[test_index]
    dt_clf.fit(x_train,y_train)
    pred=dt_clf.predict(x_test)
    accuracy=np.round(accuracy_score(y_test,pred),4)
    n_iter+=1
    train_size=x_train.shape[0]
    test_size=x_test.shape[0]
    print(n_iter,'번째 fold','교차 검증 정확도 : ',accuracy,',훈련데이터 사이즈 : ',train_size,'.테스트 데이터 사이즈 : ',test_size)
    cv_accuracy.append(accuracy)
cv_accuracy
np.mean(cv_accuracy)

### Stratified k 폴드  => 불균형한 분포도를 가진 레이블 데이터 집합을 위한 k 폴드 방식 ( 데이터를 나눌때 전체 레이블 값의 분포도를 반영함 )
from sklearn.model_selection import StratifiedKFold
skf=StratifiedKFold(n_splits=3)
n_iter=0

for train_index,test_index in skf.split(features,label):
    n_iter+=1
    label_train,label_test=label[train_index],label[test_index]
    print(pd.Series(label_train).value_counts())
    print(pd.Series(label_test).value_counts())

### cross_val_score()  => 교차검증을 편리하게 수행
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.datasets import load_iris

iris_data=load_iris()
dt_clf=DecisionTreeClassifier(random_state=156)
data=iris_data.data
label=iris_data.target
scores=cross_val_score(dt_clf,data,label,scoring='accuracy',cv=5,n_jobs=5)

np.mean(scores)
scores=cross_validate(dt_clf,data,label,scoring='accuracy',cv=5,n_jobs=5) # 테스트 데이터 정확도 뿐만아니라 수행시간과 훈련데이터 정확도 등을 알려줌

## GridSearchCV - 교차검증과 최적 하이퍼파라미터 튜닝을 한번에
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
grid_parameters={'max_depth':[1,2,3],'min_samples_split':[2,3]}
dtree=DecisionTreeClassifier()
train_x,test_x,train_y,test_y=train_test_split(features,label,test_size=0.2,random_state=124)
grid_tree=GridSearchCV(dtree,grid_parameters,scoring='accuracy',cv=3,n_jobs=5,refit=True,iid=False) # refit은 최적의 하이퍼 파라미터를 찾앗을 경우 그 파라미터로 재학습 시켜 모델 남김
grid_tree.fit(train_x,train_y)
### 파라미터에 따른 cv 결과 그래프로 그리기 => grid 서치를 어느 방향으로 해야하는지 알아야 하므로
grid_results=pd.DataFrame(grid_tree.cv_results_)
grid_results
scores=np.array(grid_results.mean_test_score.sort_values(ascending=False)).reshape(3,2)
scores
fig,ax=plt.subplots()
im=ax.imshow(scores,cmap='PiYG')
cbar=ax.figure.colorbar(im,ax=ax)
cbar.ax.set_ylabel(ylabel='max_depth',rotation=-90,va="bottom")
ax.set_yticks(np.arange(len(set(grid_results.param_max_depth))))
ax.set_xticks(np.arange(len(set(grid_results.param_min_samples_split))))
ax.set_xticklabels(grid_parameters['min_samples_split'])
ax.set_yticklabels(grid_parameters['max_depth']) # => max_depth가 작은쪽으로 해야함 => 1이 최대 작으므로 여기서 마무리

grid_tree.best_estimator_ # 최적의 하이퍼파라미터 출력
grid_tree.best_params_ # 최적의 하이퍼파라미터 출력 / grid 한 파라미터 만 출력
x=grid_tree.cv_results_ # x 변수에서 view 로 확인하기 => 그리드서치 파라미터 cv 결과들
grid_tree.best_score_
pred=grid_tree.predict(test_x)
accuracy_score(test_y,pred)

### Nested Cross-validation (중첩 교차 검증) => 완전 모델 성능 평가 비교를 위해서
grid_tree=GridSearchCV(dtree,grid_parameters,scoring='accuracy',cv=3,n_jobs=5,refit=True,iid=False)
scores=cross_val_score(grid_tree,data,label,scoring='accuracy',cv=5,n_jobs=5) # grid_tree
np.mean(scores)

# 5. 데이터 전처리

## 데이터 인코딩

### 레이블 인코딩
from sklearn.preprocessing import LabelEncoder

items=['TV','냉장고','전자레인지','컴퓨터','선풍기','선풍기','믹서','믹서']
encoder=LabelEncoder() # labelencoder 객체 생성
encoder.fit(items) # fit과 transform 수행
labels=encoder.transform(items)
print(labels)
encoder.classes_ # 어떤 문자가 어떤 숫자에 인코딩 됐는지 알 수 있음
#### ML 에서 factor 지정이 힘드므로 이런 인코딩된게 숫자형으로 쓰였을때 크고 작음이 있기에 문제가 발생 => 원핫인 코딩 필요!

### 원핫인 코딩
#### 주의점 : 1. 변환하기 전에 모든 문자열 값이 숫자형 값으로 변환돼야한다 / 2. 입력값으로 2차원 데이터가 필요하다는점
from sklearn.preprocessing import OneHotEncoder
items=['TV','냉장고','전자레인지','컴퓨터','선풍기','선풍기','믹서','믹서']
encoder=LabelEncoder()
encoder.fit(items)
labels=encoder.transform(items) # 변환 전 숫자형 값으로 변환
labels=labels.reshape(-1,1) # 2차원 데이터로 변환
type(labels)
oh_encoder=OneHotEncoder()
oh_encoder.fit(labels)
oh_labels=oh_encoder.transform(labels)
print(oh_labels.toarray())
oh_labels.shape
##### 더 편하게 원핫인코딩을 할 수 있게 하는 api 있음 => get_dummies() / 숫자형 변환도 필요없음
df=pd.DataFrame({'item':['TV','냉장고','전자레인지','컴퓨터','선풍기','선풍기','믹서','믹서']})
pd.get_dummies(df)
df_1=pd.get_dummies(df['item'])
df_2=pd.concat([df,df_1],axis=1)

## 피처 스케일링과 정규화

### StandardScaler - 표준화(표준정규분포)를 쉽게 지원하기 위한 클래스
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
iris=load_iris()
iris_data=iris.data
iris_df=pd.DataFrame(iris_data,columns=iris.feature_names)
iris_df.mean()
iris_df.var()                     # 평균과 분산이 제각각이다 => 표준화 해주기
scaler=StandardScaler()
scaler.fit(iris_df)
iris_ss=scaler.transform(iris_df)
iris_ss                           # scle transform 하면 array 형태로 출력 => dataframe 형태로 변환 필요
iris_df_scaled=pd.DataFrame(iris_ss,columns=iris.feature_names)
iris_df_scaled.mean()             # 거의 0 => 평균 0 분산 1인 정규분포 형태로 만들기때문에
iris_df_scaled.var()

### MinMaxScaler  - 정규화 / 데이터 값을 0 과 1 사이의 범위 값으로 변환 (음수가 있으면 -1에서 1 사이의 값으로 변환) => 데이터가 가우시안분포가 아닌경우 이용
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(iris_df)
iris_scaled=scaler.transform(iris_df)
iris_df_scaled=pd.DataFrame(iris_scaled,columns=iris.feature_names)
iris_df_scaled.min()
iris_df_scaled.max()

### 학습데이터와 테스트 데이터의 스케일링 변환시 유의점
#### 학습데이터에서 fit 함수 쓴걸로 테스트 데이터 transform 해야됨 => 테스트 데이터에서 다시 fit 하면 새로운 기준으로 적용돼서 그렇게 하면 안됨
#### 가장 좋은거는 전체 데이터의 스케일링 변환을 하고 난 다음에 학습과 테스트 데이터 분리

# 6. 타이타닉 생존자 예측

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

titanic_df = pd.read_csv('C:/Desktop/Son/titanic/train.csv',engine='python')
titanic_df.info()
titanic_df.head(3)
titanic_df['Cabin'].value_counts()
titanic_df['Embarked'].value_counts()

## 결측값 처리
titanic_df['Age'].fillna(titanic_df['Age'].mean(),inplace=True)
titanic_df['Cabin'].fillna('N',inplace=True)
titanic_df['Embarked'].fillna('N',inplace=True)
titanic_df.isnull().sum()
titanic_df.isnull().sum().sum()

## 변수 전처리
titanic_df['Sex'].value_counts()
titanic_df['Cabin'].value_counts()    ## 속성값들이 너무 제각각 => 앞의 선실 등급이 중요한 키워드이기때문에 그것만 살리면 좋을듯
titanic_df['Embarked'].value_counts()
titanic_df['Cabin']=titanic_df['Cabin'].apply(lambda x : x[0]) # 또는 titanic_df['Cabin']=titanic_df['Cabin'].str[:1]

titanic_df['Survived'].value_counts()
titanic_df.groupby(['Sex','Survived'])['Survived'].count() # 차이가 극명함 => survived 예측에 sex는 좋은 변수
sns.barplot(x='Sex',y='Survived',data=titanic_df) # 생존율로 막대그래프 그려짐

sns.barplot(x='Pclass',y='Survived',data=titanic_df) # 부자와 가난한 사람의 생존확률을 확인하기 위해 부를 측정 할 수 있는 속성인 객실 등급으로 시각화
sns.barplot(x='Pclass',y='Survived',hue='Sex',data=titanic_df) # hue로 분류 파라미터 추가!! / 성별과 함께 고려해서 분석하니 효율적! => 여성의 경우 삼등실 , 남성의 경우 이,삼등실의 생존확률 확연히 낮음

def get_category(age):         # 나이에 따른 생존율 확인
    cat=''
    if age<=-1: cat='Unknown'
    elif age<=5: cat='Baby'
    elif age<=12: cat='Child'
    elif age<=18: cat='Teenager'
    elif age<=25: cat='Student'
    elif age<=35: cat='Young Adult'
    elif age<=60: cat='Adult'
    else : cat='Elderly'
    return cat
plt.style.use('ggplot')
plt.figure(figsize=(10,6)) # 더 크게 보기위해
titanic_df['Age_cat']=titanic_df['Age'].apply(lambda x: get_category(x))
titanic_df.groupby(['Age_cat','Sex','Survived'])['Survived'].count()
group_names=['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Elderly']
sns.barplot(x='Age_cat',y='Survived',hue='Sex',data=titanic_df,order=group_names)            # order을 통해서 barplot 순서 정할수 있다!
sns.barplot(x='Age_cat',y='Survived',hue='Pclass',data=titanic_df,order=group_names)
titanic_df.drop('Age_cat',axis=1,inplace=True)
#### => 이런 그래프들을 보면 sex,age,plcass 등이 생존에 중요한 피쳐임을 알 수 있다
