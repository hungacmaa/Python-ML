import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
# ------------------ C L E A N I N G     D A T A ----------------------

data0 = pd.read_csv("lop70.csv")
data1 = pd.read_csv("lop71.csv")
data2 = pd.read_csv("lop72.csv")
data0.columns = ["name", "cc", "kt", "tl", "outcome"]
data1.columns = ["name", "cc", "kt", "tl", "outcome"]
data2.columns = ["name", "cc", "kt", "tl", "outcome"]

data = pd.concat([data0,data1,data2]) # nối tập dữ liệu
del[data["name"]] # xóa cột name vì không có tác dụng
so_sv = data.shape[0]
data.index = range(1, so_sv+1) #sửa lại index

data.info()

data = data.replace(0, np.NaN)
print(data.eq(0).sum())
print(data.isnull().sum())
print((data.isnull().sum()/so_sv)*100) # phần trăm ô null

data.fillna(data.mean(numeric_only=True), inplace = True) # replace NaN with the mean

data.to_csv("diemtk.csv", index = False)

# ------------------------------------------------------------------------

# ------------------ F E A T U R E S      S E L E C T I O N ----------------------
corr = data.corr()
print(corr)
# Nhận xét:
#     Ta thấy mức độ ảnh hưởng của các điểm cc -> kt -> tl đến điểm thi tăng dần
#     Điều đó là hoàn toàn hợp lý
# --------------------------------------------------------------------------------

# -------------------- V I S U A L I Z A T I O N ---------------------------
plt.subplot(3, 1, 1)
plt.plot(data["cc"])
plt.xlabel("sinh vien")

plt.subplot(3, 1, 2)
plt.plot(data["kt"])
plt.xlabel("sinh vien")

plt.subplot(3, 1, 3)
plt.plot(data["tl"])
plt.xlabel("sinh vien")

plt.show()

# Biểu đồ tần suất

plt.hist(data["cc"])
plt.xlim(0, 10)
plt.ylim(0, 100)
plt.xlabel("diem")
plt.ylabel("so lan xuat hien")
plt.title("bieu do tan suat diem cc")
plt.show()

plt.hist(data["kt"])
plt.xlim(0, 10)
plt.ylim(0, 100)
plt.xlabel("diem")
plt.ylabel("so lan xuat hien")
plt.title("bieu do tan suat diem kt")
plt.show()

plt.hist(data["tl"])
plt.xlim(0, 10)
plt.ylim(0, 100)
plt.xlabel("diem")
plt.ylabel("so lan xuat hien")
plt.title("bieu do tan suat diem tl")
plt.show()

plt.hist(data["outcome"])
plt.xlim(0, 10)
plt.ylim(0, 100)
plt.xlabel("outcome")
plt.ylabel("so lan xuat hien")
plt.title("bieu do tan suat diem thi")
plt.show()

# biểu đồ phân tán
plt.scatter(data["cc"], data["outcome"])
plt.xlabel("diem cc")
plt.ylabel("diem thi")
plt.show()

plt.scatter(data["kt"], data["outcome"])
plt.xlabel("diem kt")
plt.ylabel("diem thi")
plt.show()

plt.scatter(data["tl"], data["outcome"])
plt.xlabel("diem tl")
plt.ylabel("diem thi")
plt.show()


# --------------------- X A Y     D U N G     M O D E L -----------------------
X = data[['cc','kt','tl']].values
y = data.iloc[:,3].values

x_train, x_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.09, random_state=5)

from sklearn import linear_model

model = linear_model.LinearRegression()
model.fit(x_train, Y_train)
print("diem cua bo train: "+str(model.score(x_train, Y_train)))
print("diem cua bo test: "+str(model.score(x_test, Y_test)))
# print(model.intercept_)
# print(model.coef_)
print(model.predict([[2, 2, 2]]))


# poly
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

X = data[['cc','kt','tl']].values
y = data.iloc[:,3].values

# chot
degree = 3
poly_features = PolynomialFeatures(degree = degree)
X_poly = poly_features.fit_transform(X)
print(X_poly.shape)

x_train, x_test, Y_train, Y_test = train_test_split(X_poly, y, test_size = 0.09, random_state=5)

model = linear_model.LinearRegression()
model.fit(x_train, Y_train)
print("diem cua bo train: "+str(model.score(x_train, Y_train)))
print("diem cua bo test: "+str(model.score(x_test, Y_test)))
# print(model.intercept_)
# print(model.coef_)
check = [[2,2,2]]
check = poly_features.fit_transform(check)
model.predict(check)
