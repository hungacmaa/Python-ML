# --------------- N U M P Y -------------------------
import numpy as np #khai bao thu vien

a = np.array([1, 2, 3]) #khoi tao mang
print(a) # [1 2 3] 

print(type(a))  # kieu du lieu
#<class 'numpy.ndarray'>

np.zeros((3,4), dtype = int) #Tạo mảng hai chiều các phần tử 0 với kích thước 3x4.
# array([[0, 0, 0, 0],
#        [0, 0, 0, 0],
#        [0, 0, 0, 0]])

np.ones((2,3,4), dtype = int) #Tạo mảng 3 chiều các phần tử 1 với kích thước 2x3x4.
# array([[[1, 1, 1, 1],
#         [1, 1, 1, 1],
#         [1, 1, 1, 1]],

#        [[1, 1, 1, 1],
#         [1, 1, 1, 1],
#         [1, 1, 1, 1]]])

np.arange(1,7,2) #Tạo mảng 1 chiều với các phần tử từ 1 - 6 với bước nhảy là 2.
# Out[8]: array([1, 3, 5])

np.full((2,3),5) #Tạo mảng 2 chiều các phần tử 5 với kích thước 2x3.
# array([[5, 5, 5],
#        [5, 5, 5]])

np.eye(4, dtype=int) #Tạo ma trận đơn vị với kích thước là 4x4, các phần từ ở đường chéo từ trái sang phải là 1, còn lại là 0
# array([[1, 0, 0, 0],
#        [0, 1, 0, 0],
#        [0, 0, 1, 0],
#        [0, 0, 0, 1]])

np.random.random((2,3)) #Tạo ma trận các phần tử ngẫu nhiên với kích thước 2x3. Các giá trị random từ 0-1
# array([[0.40303507, 0.12155649, 0.29515914],
#        [0.14391111, 0.43014381, 0.54877088]])


# Cac thuoc tinh cua numpy
print("Kiểu dữ liệu của phần tử trong mảng:", a.dtype) #Kiểu dữ liệu của phần tử trong mảng: int32
print("Kích thước của mảng:", a.shape) #Kích thước của mảng: (3,)
print("Số phần tử trong mảng:", a.size) #Số phần tử trong mảng: 3
print("Số chiều của mảng:", a.ndim) #Số chiều của mảng: 1

print(a[a % 2==1]) #Boolean indexing
# [1 3]

# np.loadtxt('ten_file', dtype = int, delimiter=','): doc file

b = np.array([[1,2],
              [3,4]])
np.median(b)

a.reshape(-1, 1)
a.reshape(1, -1)
