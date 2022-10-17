import numpy as np

a = np.array([1, 2, 3])
print(a) # [1 2 3]

print(type(a)) #<class 'numpy.ndarray'>

a1 = np.zeros((3,4), dtype = int)
print(a1)

a2 = np.ones((2,3,4), dtype = int)
print(a2)

a3 = np.arange(1,7,2)
print(a3)

a4 = np.full((2,3),5)
print(a4)

a5 = np.eye(4, dtype=int)
print(a5)

a6 = np.random.random((2,3))
print(type(a6))

print("Kiểu dữ liệu của phần tử trong mảng:", a.dtype)
print("Kích thước của mảng:", a2.shape)
print("Số phần tử trong mảng:", a2.size)
print("Số chiều của mảng:", a2.ndim)

print(a[[2, 1, 0]])

#Khởi tạo mảng một chiều với kiểu dữ liệu các phần tử là Integer
arr = np.array([1,3,4,5,6], dtype = int)

#Khởi tạo mảng một chiều với kiểu dữ liệu mặc định
arr = np.array([1,3,4,5,6])
print(arr)

arr1 = np.array([(4,5,6), (1,2,3)], dtype = int)
print(arr1)

arr2 = np.array(([(2,4,0,6), (4,7,5,6)],
                 [(0,3,2,1), (9,4,5,6)],
                 [(5,8,6,4), (1,4,6,8)]), dtype = int)
print(arr2)


print("arr[2]=", arr[2])
print("arr1[1,2]=", arr1[1,2])
print("arr2[1,2,3]=", arr2[1,1,3])
print("arr[0:3]=", arr[0:3])
print("arr1[:,:1]=", arr1[:,:2])

np.loadtxt('Diem_2A.txt', dtype = int, delimiter=',') #ở đây tất cả phần tử là số nguyên nên mình để kiểu int cho dễ nhìn, các phần tử phân tách nhau bởi dấu ","

print(arr1.sum())
print(arr1.mean())
print(np.median(arr))
print(arr[arr % 2==1])
print(a.reshape(-1, 1))

x = np.array([2,3])
y = np.array([4,2])
print(np.dot(x,y)) # 2x4 + 3x2 = 14

x2 = np.matrix([[1,2],[4,5]])
y2 = np.matrix([[7,8],[2,3]])
print(type(x2))

a = np.array([(1,2,3),(4,5,6),(7,8,9)])
print(a)
print(a.cumsum(axis=0)) # axis = 0 la tinh theo chieu doc
print(a.cumsum(axis=1)) # axis = 1 la tinh theo chieu ngang

a = np.array([3,2,1])
print(a.cumsum(axis=1))
print(a.argsort())

print(a.sum(axis=1))
'''
[[1 2 3]
[4 5 6]
[7 8 9]]
'''