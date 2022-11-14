# --------------- P A N D A S -------------------------
import pandas as pd 
import numpy as np

s = pd.Series([0,1,2,3])
print(s)

# 0    0
# 1    1
# 2    2
# 3    3

s = pd.Series([0,1,2,3], index=["a","b","c","d"])

data = {'a' : -1.3, 'b' : 11.7, 'd' : 2.0, 'f': 10, 'g': 5}
ser = pd.Series(data,index=['a','c','b','d','e','f'])
print(ser)

ser = pd.Series(5, index=[1, 2, 3, 4, 5])
print(ser)

data = {'a' : -1.3, 'b' : 11.7, 'd' : 2.0, 'f': 10, 'g': 5}
ser = pd.Series(data,index=['a','c','b','d','e','f'])
print(ser['d'])
print(ser['c'])

print(ser[:'d'])

b = np.asarray(ser)
print(b)
print(type(b))

dates1 = pd.date_range('20190525', periods=12)
print(dates1)

dates1 = pd.date_range(start='19991204', periods=12, freq="MS")
print(dates1)

series = pd.Series([1,2,3,4,5,6,7,8,9,10,11,12])
series.index = dates1
print(series)

s = {'một': pd.Series([1., 2., 3., 5.], index=['a', 'b', 'c', 'e']),
     'hai': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}

# tạo DataFrame từ dict
df = pd.DataFrame(s)

df = pd.DataFrame(s, index=['a','c','d'])
print(df)

df_hai = df['hai']
print(df_hai)

df['ba'] = [7,8,9]
df['bốn'] = df['hai'] - df['ba']

df['Chuẩn'] = 'OK'

df['Khác'] = df['hai'][:3]

df['KT'] = df['một'] == 3.0

df.insert(2, 'chèn', df['một'])

data_excel = pd.read_excel('diem.xlsx')

data_excel.describe()

data_excel.head()

print(data.sort_index(axis=1))

print(data.sort_values(by="Name"))

data_excel.drop(['Name'], axis=1)

data_excel.to_csv('diem.csv')

print(data_excel.iloc['name'])