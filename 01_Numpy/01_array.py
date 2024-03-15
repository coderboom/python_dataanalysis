import numpy as np

print(np.__version__)
"""
numpy.array(object, dtype=None, copy=True, order='K', subok=False, ndmin=0)

参数说明：
object：必填参数，可以是一个序列（如列表、元组或嵌套序列），或者其他具有 array-like 特性的对象，或者是实现了 array 方法的对象。
    这个参数是要转换成数组的数据源。
dtype：可选参数，指定数组元素的数据类型，默认情况下会根据输入数据自动推断类型。
    如果提供了具体的 dtype，则会尝试将输入数据转换为指定的数据类型。
copy：布尔型参数，决定是否需要复制数据。
    默认为 True，即总是创建一个新的数组；若设置为 False，且输入数据是可被视作数组的对象（如已经存在的 ndarray），
    则返回与原数据共享内存的视图（view）。
order：字符参数，控制数组内存布局：
    'C' 表示行主序（C-style contiguous），这是大多数情况下数组的默认顺序。
    'F' 表示列主序（Fortran-style contiguous）。
    'A' 表示任何顺序，优先选择已有的连续布局。
    'K'（默认值）表示保持输入数据的原有顺序。
subok：布尔型参数，当为 True 时，允许返回与输入具有相同类型的子类数组。默认为 False，意味着始终返回基类 ndarray。
ndmin：整数参数，规定生成数组的最小维度。如果输入数据的维数小于指定值，将在前面添加适当的轴以满足要求。


整数类型：
numpy.int8, numpy.int16, numpy.int32, numpy.int64：分别对应8位、16位、32位和64位有符号整数。
numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64：分别对应8位、16位、32位和64位无符号整数。
浮点数类型：
numpy.float16, numpy.float32, numpy.float64: 分别代表半精度（16位）、单精度（32位）和双精度（64位）浮点数。
numpy.float_: 默认为平台相关的浮点数精度，通常等于numpy.float64。
复数类型：
numpy.complex64, numpy.complex128: 分别表示由两个32位浮点数和两个64位浮点数组成的复数。
numpy.complex_: 默认的复数类型，通常等同于numpy.complex128。
布尔类型：
numpy.bool_: 与Python的bool类型一致，但可以存储在数组中。
其他类型：
numpy.object_: 可以存储任意Python对象。
numpy.string_ 和 numpy.unicode_: 字符串类型，后者用于Unicode字符。
numpy.datetime64 和 numpy.timedelta64: 用于表示日期和时间间隔的类型。
类似还有结构化数据类型（structured arrays），可以通过numpy.dtype定义包含多个字段的数据类型。
"""

# 从列表创建一个一维数组
a = np.array([1, 2, 3, 4])
print(a, a.shape)  # 输出：[1 2 3 4]
a1=a.T # 一维数组没有转置的概念
print(a1.T) 

# 创建一个二维数组，同时指定数据类型
b = np.array([(1.5, 2, 3), (4, 5, 6)], dtype=np.float32)
print(b)
# 输出：
# [[1.5 2.  3. ]
#  [4.  5.  6. ]]

b1 = np.array(['abcda', 'qiweiyu', 'adhiqhw'], dtype=np.string_)
print(b1)
b2 = np.array(['abcda', 'qiweiyu', 'adhiqhw'], dtype=np.unicode_)
print(b2)
"""
输出： ['abcda', 'qiweiyu', 'adhiqhw'] <U6
大端字节序：内存地址从高到低
小端字节序：内存地址从低到高
"""

# 使用 copy 参数创建视图而非副本
c = np.array([[1, 2], [3, 4]])
d = np.array(c, copy=False)
""" 
# 现在修改 d 会影响到 c，因为它们共享同一块内存区域,
# ————是的，这是浅复制。通过将copy参数设置为False，d成为了c的一个视图，它们共享相同的内存区域。
# 因此，对d所做的任何修改都会影响到c。如果想要创建一个独立的副本，需要将copy参数设置为True，即d = np.array(c, copy=True)。
"""

# 使用 ndmin 参数，确保至少为二维数组
e = np.array([1, 2, 3], ndmin=2)
print(e.shape)  # 输出：(1, 3)
# 二维数组的样式[[1,2,3]] 这是一个二维数组，但这里只是1行3列

"""
ndmin参数指定了返回的数组应该具有的最小轴数。在上述代码中，将ndmin设置为2，表示返回的数组至少应该有两个轴。
 因此，当使用np.array([1, 2, 3], ndmin=2)创建数组e时，虽然输入的是一维数组[1, 2, 3]，
 但返回的数组e会是一个二维数组，其中第一维的长度为1，第二维的长度为3，即e.shape的输出为(1, 3)。

在NumPy中，轴（axis）是描述多维数组结构的一个重要概念。一个多维数组的每个维度都有一个对应的轴。轴可以理解为数组的一个方向或索引顺序。
例如，在二维数组（矩阵）中，通常有两条轴：
轴0（axis=0），也称为行轴，对应于矩阵中的每一行。
轴1（axis=1），也称为列轴，对应于矩阵中的每一列。
对于更高维度的数组，如三维数组，就有三个轴：
轴0仍然是最外层，可能对应于一组二维矩阵；
轴1则是这些二维矩阵内部的行；
轴2则是这些二维矩阵内部的列。
简而言之，在NumPy数组中，轴就是数组各个维度的方向，它允许我们沿着特定维度进行操作，
比如切片、拼接、转置以及进行各种数学运算等。当调用某些NumPy函数时，通过指定axis参数，我们可以控制函数是在哪个维度上执行操作。
"""
# 使用 dtype 和 order 参数
f = np.array([[1, 2], [3, 4]], dtype=np.int8, order='F')
print(f)
# [[1 2]
# [3 4]]

print('------------------arange()---------------')
"""
pd.arange()是Pandas库中的一个函数，用于创建一个Index对象，其值从指定的起始值开始，以一定的步长递增，直到指定的结束值（不包含）。
    它与Python标准库中的range()函数类似，但返回的是一个Index对象而不是Python的内置range对象。

以下是pd.arange()函数的一些重要参数：
start：起始值，默认为0。
stop：结束值，生成的Index对象将不包含此值。
step：步长，默认为1。
dtype：Index对象中元素的数据类型，默认为int64。
"""

# 创建一个从0开始，到9结束，步长为2的Index对象
index = np.arange(start=0, stop=10, step=2)
print(index)

print('------------------linspace()---------------')
"""np.linspace()是一个NumPy函数，用于创建一个等差数列。它会生成一个指定范围内均匀分布的数组。

以下是np.linspace()函数的一些重要参数：
    start：数列的起始值。
    stop：数列的结束值(包含)。
    num：数列中的元素个数，默认为10。
    endpoint：是否包含数列的结束值，默认为True。
    retstep：是否返回数列的步长，默认为False——设置为true后，函数返回的是一个二元元组（数组，步长）。
    dtype：生成数组的数据类型，默认为float64。
"""
# 创建一个包含10个元素，起始值为0，结束值为1的等差数列
arr = np.linspace(start=0, stop=1, num=10, retstep=True)
print(arr)

print('------------------logspace()---------------')
"""
np.logspace()是NumPy库中的一个函数，用于创建等比数列（对数空间中的线性分布），生成的数组元素在对数尺度上是均匀分布的。

以下是np.logspace()函数的重要参数：
    start：数列中第一个元素的底数为10的对数值。
    stop：数列中最后一个元素的底数为10的对数值。
    num：数列中元素的个数，默认为50。
    endpoint：是否包含stop指定的值，默认为True。
    base：对数使用的基数，默认为10。
    dtype：生成数组的数据类型，默认为float64。
"""
# 创建一个从10^0到10^2（包含）的等比数列，共5个元素
arr = np.logspace(start=0, stop=2, num=5)
print(arr)
arr1 = np.logspace(start=0, stop=4, num=10, base=2, endpoint=False)
print(arr1)

print('------------------创建二维数组---------------')
"""

"""
# 创建二维数组
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr)
arr1=arr.T
print(arr1)














































