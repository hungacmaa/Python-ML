#------------ C L E A N I N G       D A T A ------------------------------
import numpy as np
import pandas as pd

'''
Dữ liệu chuẩn bị:
  A     B       C
0 1     2.0     3
1 4     NaN     6
2 7     NaN     9
3 10    11.0    12
4 13    14.0    15
5 16    17.0    18
'''
NaN_dataset = pd.read_csv("NaNDataset.csv")
print(NaN_dataset)

# Xem các giá trị null
NaN_dataset.isnull().sum()

# Loại bỏ các hàng null
NaN_dataset_after_dropna = NaN_dataset.dropna()
print(NaN_dataset_after_dropna)
# reset index
NaN_dataset_after_dropna = NaN_dataset_after_dropna.reset_index()
print(NaN_dataset_after_dropna)

# Thay thế các null bằng giá trị trung bình của cột
NaN_dataset_fillna_by_mean = NaN_dataset.iloc[:]
NaN_dataset_fillna_by_mean['B'] = NaN_dataset_fillna_by_mean['B'].fillna(NaN_dataset_fillna_by_mean['B'].mean())
print(NaN_dataset_fillna_by_mean)

Duplicate_dataset = pd.read_csv("DuplicateRows.csv")
print(Duplicate_dataset)
print(Duplicate_dataset.duplicated(keep="first"))

print(Duplicate_dataset.drop_duplicates(keep = "first", # Giữ lại giá trị trùng lặp đầu tiên
                                        inplace=False)) # Không thay đổi tập dữ liệu gốc

# Chuẩn hóa dữ liệu
Nomalize_dataset = pd.read_csv("NormalizeColumns.csv")
print(Nomalize_dataset)

from sklearn import preprocessing
x = Nomalize_dataset.values.astype(float) # lấy mảng dữ liệu (numpy) đưa về kiểu float
scaler = preprocessing.MinMaxScaler() # gọi đối tượng min_max scaler
x = scaler.fit_transform(x) 
new_Nomalize_dataset = pd.DataFrame(x, columns = Nomalize_dataset.columns)
print(new_Nomalize_dataset)

# Xử lý outlier
import matplotlib.pyplot as plt
Outlier_dataset = pd.DataFrame({"Point":[1,4,2,5,10,15,9999,0]})
Outlier_dataset.Point.plot(kind="box")
plt.show()

Outlier_dataset = Outlier_dataset.loc[Outlier_dataset.Point<5000]
Outlier_dataset = Outlier_dataset.reset_index(drop=True)
print(Outlier_dataset)

