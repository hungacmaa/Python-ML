import pandas as pd
import numpy as np
s = pd.Series([0,1,2,3])
print(s.index)

s = pd.Series([0,1,2,3], index=["a","b","c","d"])
print(s.index)

data = {'a' : -1.3, 'b' : 11.7, 'd' : 2.0, 'f': 10, 'g': 5}
ser = pd.Series(data,index=['a','c','b','d','e','f'])
print(ser)

ser = pd.Series(5, index=[1, 2, 3, 4, 5])
print(ser)

data = {'a' : -1.3, 'b' : 11.7, 'd' : 2.0, 'f': 10, 'g': 5}
ser = pd.Series(data,index=['a','c','b','d','e','f'])

print(ser['d'])
print(ser['g'])

print(ser[:'d'])


b = np.asarray(ser)
print(b)
print(type(b))

dates1 = pd.date_range(start='05/05/2001', periods=12, freq="MS")
print(dates1)
series = pd.Series([1,2,3,4,5,6,7,8,9,10,11,12])
series.index = dates1
print(series)
print(series.index.dtype)

# tạo dict từ các series
s = {'một': pd.Series([1., 2., 3., 5.], index=['a', 'b', 'c', 'e']),
     'hai': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}

# tại DataFrame từ dict
df = pd.DataFrame(s)
print(df)

df = pd.DataFrame(s, index=['a','c','d'])
print(df)

df_hai = df['hai']
print(df_hai)

# thêm cột bốn với giá trị mỗi ô theo công thức
df['ba'] = df['hai'] - df['một']

# thêm cột với giá trị vô hướng (scalar value)
df['Chuẩn'] = 'OK'

# thêm cột không cùng số lượng index với DataFrame
df['Khác'] = df['hai'][:3]

# thêm cột True/False theo điều kiện
df['KT'] = df['một'] == 3.0

# dùng hàm insert. Cột "chèn" bên dưới sẽ ở vị trí 2 (tính từ 0), có giá trị bằng cột một
df.insert(2, 'chèn', df['một'])

print(df)

print((df['một'] == 3.0 and df['hai'] == 3.0))
