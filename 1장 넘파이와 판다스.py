import numpy as np
import pandas as pd

# 3. 넘파이

## ndarray 개요

array1=np.array([1,2,3]) # 1차원 데이터
type(array1)
array1.shape # shape 변수는 ndarray의 크기(행과 열의 수 / 배열의 차원까지)
array2=np.array([[1,2,3],[2,3,4]])
array2.shape
array3=np.array([[1,2,3]]) # 2차원 데이터
array3.shape
array1.ndim  # ndarray의 차원 확인
array2.ndim
array2.ndim

## ndarray의 데이터 타입

array1=np.array([1,2,3.]) # int 형과 float형을 같이 써도 더큰 개념인 float로 출력됨 => ndarray는 같은 데이터타입만 됨
print(array1)
array1.dtype # 내부 타입을 확인해보면 float으로 출력됨
array2=np.array([1,2,'a'])
print(array2) # 더큰 개념인 문자로 출력됨( 자동 형변환 )
array2.dtype
array1.astype('int') # astype 을 이용하면 원하는 타입으로 변경 가능 (메모리 절약을 위해 주로 이용)

## ndarray를 편리하게 생성하기 - arange,zeros,ones

sequence_array=np.arange(10) # 1차원 ndarray 만들기 (range 처럼 만들기)
print(sequence_array)
print(sequence_array.dtype,sequence_array.shape)
array_0=np.zeros(10,dtype=int) # 1차원 0 으로 찬 ndarray 만들기
array_0
array_0.shape
array_1=np.ones(10,dtype=int) # 1차원 1로 가득 찬 ndarray 만들기
array_1

## ndarray의 차원과 크기를 변경하는 reshape

array_1.reshape(5,2) # 차원과 크기를 변경하는 reshape()
array_0.reshape(2,5)
array_1.reshape(-1,2) # 행이나 열에 -1을 지정하면 다른 열 또는 행에 맞춰서 알아서 크기 설정해줌
array_0.reshape(5,-1)
array_0.tolist() # Tolist 함수를 이용해서 ndarray를 리스트로 다시 되돌릴 수 잇다
array_2=np.arange(8)
array_2=array_2.reshape((2,2,2))
array_2.tolist()

## ndarray 인덱싱

array1=np.arange(8)
print(array1[-1],array1[-2],array1[3])
array1[5]=4;print(array1)
array1=np.arange(8).reshape(2,4)
array1[1,2]
array1[0:1,0:2]
array1[0] # 2차원에서 1차원 array 추출 가능 (행 추출 가능)
array3=np.arange(27).reshape(3,3,3)
array3[0] # 3차원에서 2차원 array 추출 가능
array1[[0,1],1] # 1~2 행의 1열 값을 추출
array1[array1>5]

## 행렬의 정렬 sort 와 argsort

org_array=np.array([3,1,9,5])
sorting=np.sort(org_array); print(sorting,org_array)  # 원본은 남겨둔 채 솔트됨
inverse_sort=np.sort(org_array)[::-1];print(inverse_sort,sorting) # 내림차순으로 정렬
array2d=np.array([8,12,7,1]).reshape(2,2)
sort_array2d_0=np.sort(array2d,axis=0);print(sort_array2d_0) # 2차원 array를 행 기준으로 sort ( 즉, 한열에 2개의 행을 sort 하는것)
sort_array2d_1=np.sort(array2d,axis=1);print(sort_array2d_1)
name_array=np.array(['john','mike','sarah','kate','samuel'])
score_array=np.array([78,95,84,98,88])
sort_indices=np.argsort(score_array) # 정렬된 행렬의 인덱스 반환 => ndarray는 데이터 프레임과 다르게 메타데이터(인덱스를 가진 데이터) 가 아니라서 2개의 array 생성이 필요 => 굉장히 유용
print(sort_indices)
print(name_array[sort_indices])

# 4. Pandas

## 판다스 시작

titanic=pd.read_csv('C:/Desktop/Son/titanic/train.csv',engine='python')
titanic.head()
titanic.shape # 데이터의 갯수와 변수 갯수 파악
titanic.info() # 데이터 변수들에 대한 정보
x=titanic.describe() # 변수들의 평균,최솟값,최댓값등 통계량
value_counts=titanic['Pclass'].value_counts();print(value_counts) # 변수값이 어떤 분포를 이루는지 가르쳐줌 (범주형 변수에 좋음) / series 일때 쓸수 있는데 앞의 방식이 시리즈로 변환하는것
titanic['Pclass'].head() # 시리즈 형태 => 인덱스값이 있고 데이터값이 출력됨

## DataFrame과 리스트,딕셔너리,넘파이 ndarray 상호변환

col_name=['col1','col2','col3']
list1=[[1,2,3],[4,5,6]]
array1=np.array(list1)
df_list1=pd.DataFrame(list1,columns=col_name)
print(df_list1)
df_array=pd.DataFrame(array1,columns=col_name)
print(df_array)
dict1={'col1':[1,2],'col2':[4,5],'col3':[5,6]}
df_dict1=pd.DataFrame(dict1)
print(df_dict1)

## 데이터 프레임을 ndarray,list,dict 로 변환

df2=df_dict1.values  # value 중요!! => 머신러닝 패키지의 입력인자에 ndarray를 이용하는 경우가 많다 => dataframe을 array로 변환해야하는 경우 多
df2
df2=df_dict1.values.tolist();print(df2) # 리스트로 변환
df3=df_dict1.to_dict('list');print(df3) # 딕셔너리로 변환

## DataFrame의 칼럼 데이터 세트 생성과 수정

titanic['age_0']=0 # 새로운 칼럼 추가 1
titanic.head(3)
titanic['Family']=titanic['SibSp']+titanic['Parch']+1 # 새로운 칼럼 추가 2
titanic.head(3)
titanic['Family']=titanic['Family']*2 # 기존 칼럼 업데이터
titanic.drop(['age_0','Family'],axis=1,inplace=True) # 칼럼 삭제하기
titanic.head(3)

## Index 개체

indexes=titanic.index # 인덱스 추출
print(indexes)
indexes.values # 인덱스를 array로 변환
indexes[0]=5 # Error ! 인덱스 객체는 함부로 변경할수 없음
series_fair=titanic['Fare'] # series는 index를 가지지만 연산함수에서는 index는 제외됨
series_fair.min()
series_fair.max()
series_fair.sum()
x=series_fair+3; print(x.head())
titanic_index=titanic.reset_index(inplace=False) # 인덱스를 새롭게 할당
titanic_index.head() # 기존 인덱스가 변수로 변환됨
value_counts=titanic['Pclass'].value_counts() # 시리즈로 출력
x=value_counts.reset_index(inplace=False);print(x) # 데이터 프레임으로 출력

## 데이터 셀렉션 및 필터링
### DataFrame 바로 뒤에 나오는 []는 칼럼 지정 연산자 / 인덱싱기능(인덱스 하나는 불가)
titanic[['Pclass','Age']].head(4)
titanic[0] # Error []안에 숫자 인덱스는 오류발생
titanic[0:2] # 이런건 가능
titanic[titanic['Age']==32].head(3) # 데이터 셀렉션에 유용
data={'Name':['Chulmin','Eunkyung','Heeheon','Jaewon'],'Year':[2011,2016,2015,2015],'Gender':['Male','Female','Male','Female']}
data_df=pd.DataFrame(data,index=['one','two','three','four']);print(data_df)
### iloc은 위치기반 인덱싱 => integer 형태로 입력 / loc은 명칭 기반 인덱싱 => 명칭으로 입력
titanic.iloc[0,3]
titanic.loc[0,"Name"] # 인덱스가 숫자로 되어있으므로 0이 명칭임!
titanic.loc[:3,'Name'] # 중요!!!! => 인덱스를 숫자로 받는게 아니라 명칭으로 받기때문에 '0:3' 으로 하면 0,1,2 인덱스를 출력하는게 아니라 그 자체로 0,1,2,3을 출력함
titanic[titanic['Age']>60].head(3)
titanic[titanic['Age']>60][['Name','Pclass']].head(3)
titanic.loc[titanic['Age']>60,['Name','Pclass']].head(3) # loc을 이용해도 좋고 그냥 []를 이용해도 됨
titanic.loc[(titanic['Age']>60) & (titanic['Pclass']!=1),:].head(3)
cond1=titanic['Age']>60
cond2=titanic['Pclass']!=1
titanic.loc[cond1 & cond2,:].head(3)

## 정렬,Aggregation 함수,GroupBy 적용

titanic_sorted=titanic.sort_values(by=['Name'],axcending=True,inplace=False) # 데이터 프레임 정렬하기(Sort_Value) / by로 어떤 행을 기준으로 정렬할지 설정
titanic_sorted1=titanic.sort_values(by=['Pclass','Name'],axcending=False) # 두가지 기준으로 정렬하기
titanic.count() # 변수 데이터 수 알려줌 (각 변수마다 Nan 수 찾아내기 좋음)
titanic[['Age','Fare']].mean()
titanic_group=titanic.groupby(by='Pclass') # 중요! Groupby 에서 by로 기준을 정하여 데이터를 나눔
titanic_group.count() # Pclass 3개의 범주로 나눠져서 count 된 결과가 나옴
titanic_group.head(3)
titanic.groupby(by='Pclass')[['PassengerId','Survived']].count()
titanic.groupby(by='Pclass')['Age'].agg([min,max])
agg_format={'Age':'max','SibSp':'sum','Fare':'mean'} # aggregate를 설정하여 각 변수마다 다른 통계량 추출
titanic_group.agg(agg_format)

## 결손 데이터 처리하기

titanic.isna() # Isna 는 Nan 데이터를 True 아니면 False 로 반환하여 결측치를 찾음
titanic.isna().sum() # 각 변수별 결측치 수
titanic['Age']=titanic['Age'].fillna(titanic['Age'].mean()) # 결측값 채우기 => Fillna
titanic['Age'].fillna(titanic['Age'].mean(),inplace=True)
titanic['Embarked'].fillna('S',inplace=True)

## Apply , Lambda 식으로 데이터 가공

def squaress(a) :
    return a**2
squaress(3)
squaress_1=lambda x : x**2 # Lambda 를 이용하면 한줄로 함수 요약 가능
print(squaress_1(3))
a=[1,2,3]
squares=map(lambda x : x**2,a) # map 을 통해 여러 입력인자의 경우 여러 출력값을 낼 수 있음
list(squares)
titanic['Name len']=titanic['Name'].apply(lambda x : len(x)) # Apply와 Lambda 를 이용해 데이터프레임 안에 변수를 디테일 하게 다룰 수 있다
titanic[['Name','Name len']].head(3)
titanic['Child_Adult']=titanic['Age'].apply(lambda x : 'Child' if x <=15 else 'Adult') # 주의점: lambda 에서 ifelse 를 쓸땐 if 식보다 반환값을 먼저 기술해야함 else 는 나중에 기술
titanic[['Age','Child_Adult']].head(8)  # ↓↓↓↓ ifelse 문에서 else if(elif) 는 지원 해주지 않음 => 할려면 ~ if ~ else ( ~ if ~ else ~ ) 해야함
titanic['Age_cat']=titanic['Age'].apply(lambda x : 'Child' if x<=15 else ('Adult' if x<=60 else 'Elderly'))
titanic['Age_cat'].value_counts()
### if else 문이 길어질거 같으면 따로 함수를 만들어 이용
def get_category(age):
    cat=''
    if age<=5:cat='baby'
    elif age<=12:cat='child'
    elif age<=18:cat='Teenager'
    elif age<=25:cat='Student'
    elif age<=35:cat='Young Adult'
    elif age<=60:cat='Adult'
    else :cat='Elderly'
    return cat
titanic['Age_cat']=titanic['Age'].apply(lambda x : get_category(x))