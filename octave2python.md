### 吴恩达机器学习笔记---Octave教程(Python实现)

#### 第二周 octave教程

##### 5.1节 基本操作

- 第52页

```python
import numpy as np
# 建立矩阵
A = np.mat([[1, 2], [3, 4], [5, 6]])
print("A = ", A)
V = np.mat([[1], [2], [3]])
print("V = ",  V)

# 建立特殊矩阵
A = np.ones((2, 3))
print("A = ", A)
B = np.zeros((2, 3))
print("B = ", B)
I = np.eye(6)
print("I = ", I)

C = np.random.rand(3, 3)
print("C = ", C)
D = np.random.randn(1, 3)
print("D = ", D)

# 输出
A =  [[1 2]
 [3 4]
 [5 6]]
V =  [[1]
 [2]
 [3]]
A =  [[1. 1. 1.]
 [1. 1. 1.]]
B =  [[0. 0. 0.]
 [0. 0. 0.]]
I =  [[1. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 1.]]
C =  [[0.32699556 0.12524207 0.58057881]
 [0.64533286 0.84355689 0.77487524]
 [0.48110471 0.38639891 0.80057691]]
D =  [[-0.91338325  0.72946901 -1.07624459]]
```

##### 5.2节 移动数据

- 第55页

```python
import numpy as np
import pandas as pd

# 矩阵大小
A = np.mat([[1, 2], [3, 4], [5, 6]])
print("矩阵A的大小为： ", A.shape)
print("矩阵A的行数的大小为： ", A.shape[0])
print("矩阵A的列数的大小为： ", A.shape[1])

V = np.mat([1, 2, 3, 4])
print("矩阵V的最大维度为： ", max(V.shape))

# 加载数据
path = r'C:\Users\Administrator\Desktop\ML\machine-learning-ex1\ex1\ex1data1.txt'           # 文件路径,内容为两列
data = pd.read_csv(path, header = None, names = ['data1', 'data2'])
print("加载的数据的大小为： ", data.shape)

# 从加载的数据中裁剪一部分存储到V中
V = data.iloc[:10, :]
print(V)

# 操作数据
A = np.mat([[1, 2], [3, 4], [5, 6]])
print(A[2, 1])          # 索引为（2,1）的值
print(A[1, :])          # 第二行所有元素
print(A[:, 1])          # 第二列所有元素

A[:, 1] = [[1], [2], [3]]      # 把第二列替换掉
B = [[10], [11], [12]]
C = [[10, 11]]
print(np.c_[A, B])             # 为矩阵加上列
print(np.r_[A, C])             # 为矩阵加上行

# 输出
矩阵A的大小为：  (3, 2)
矩阵A的行数的大小为：  3
矩阵A的列数的大小为：  2
矩阵V的最大维度为：  4
加载的数据的大小为：  (97, 2)
    data1    data2
0  6.1101  17.5920
1  5.5277   9.1302
2  8.5186  13.6620
3  7.0032  11.8540
4  5.8598   6.8233
5  8.3829  11.8860
6  7.4764   4.3483
7  8.5781  12.0000
8  6.4862   6.5987
9  5.0546   3.8166
6
[[3 4]]
[[2]
 [4]
 [6]]
[[ 1  1 10]
 [ 3  2 11]
 [ 5  3 12]]
[[ 1  1]
 [ 3  2]
 [ 5  3]
 [10 11]]

```

##### 5.3 计算数据

- 第63页

```python
import numpy as np
import math

A = np.mat([[1, 2], [3, 4], [5, 6]])
B = np.ones((3, 2))
C = np.mat([[1, 1], [2, 2]])

print("A*C = ", np.dot(A, C))             # 直接A*B也可以
print("A.*B = ", np.multiply(A, B))
print(A+1)

# 转置和逆
print("矩阵A的转置为： ", A.T)
print("矩阵A的逆为： ", A.I)

print("矩阵A的最大值： ", np.max(A))
print("矩阵A各元素之和为： ", np.sum(A))

# 输出
A*C =  [[ 5  5]
 [11 11]
 [17 17]]
A.*B =  [[1. 2.]
 [3. 4.]
 [5. 6.]]
[[2 3]
 [4 5]
 [6 7]]
矩阵A的转置为：  [[1 3 5]
 [2 4 6]]
矩阵A的逆为：  [[-1.33333333 -0.33333333  0.66666667]
 [ 1.08333333  0.33333333 -0.41666667]]
矩阵A的最大值：  6
矩阵A各元素之和为：  21

```

##### 5.4 绘图数据

- 第71页

```python
import numpy as np
import math
import matplotlib.pyplot as plt

t = np.arange(0, 0.98, 0.01)

y1 = np.sin(2*math.pi*4*t)
y2 = np.cos(2*math.pi*4*t)

plt.plot(t, y1)
plt.plot(t, y2)

plt.xlabel('time')                  # 横坐标
plt.ylabel('value')                 # 纵坐标
plt.legend(['sin', 'cos'])          # 标注名称
plt.title('myplot')                 # 标题

plt.show()
```

```python
import numpy as np
import math
import matplotlib.pyplot as plt

t = np.arange(0, 0.98, 0.01)

y1 = np.sin(2*math.pi*4*t)
y2 = np.cos(2*math.pi*4*t)

plt.subplot(1, 2, 1)
plt.plot(t, y1, 'r')
plt.title('sin')

plt.subplot(1, 2, 2)
plt.plot(t, y2, 'b')
plt.title('cos')

plt.show()

```

