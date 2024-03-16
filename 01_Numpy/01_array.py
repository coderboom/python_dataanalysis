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
"""在NumPy中，数组的形状（shape）是由数组的维度（dimensions）决定的。对于一维数组来说，它只有一维，没有行和列的概念，所以它的形状就是它的长度，即一个整数。"""
a = np.array([1, 2, 3, 4])
print(a, a.shape)  # 输出：[1 2 3 4]
a1 = a.T  # 一维数组没有转置的概念
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
arr1 = arr.T  # 数组转置
print(arr1)
print('-----------------ones()-------------------')
"""np.ones()是NumPy库中的一个函数，用于创建一个全为1的数组。您可以指定数组的形状，包括一维、二维乃至多维数组。
numpy.ones(shape, dtype=None, order='C')

    shape (必需参数)：指定要创建的数组的形状。它可以是一个整数（对于一维数组），也可以是一个表示多维数组各维度大小的元组，
        例如 (3, 4) 表示一个3行4列的二维数组。
    dtype (可选参数)：指定数组元素的数据类型，默认是 float64。可以设置为任何NumPy支持的数据类型，
        如 int, float32, bool 等。
    order (可选参数)：指定数组元素在内存中的存储顺序。'C'（默认）表示按行优先（C-style）存储，'F' 表示按列优先（Fortran-style）存储。
        对于一维数组和全1数组来说，这个参数通常没有影响。

# 创建一个全为1的一维数组
ones_1d = np.ones(shape, dtype=type)
# 其中shape是一个整数（例如：5）或者元组（例如：(5,)）

# 创建一个全为1的二维数组
ones_2d = np.ones(shape=(rows, cols), dtype=type)
"""

# 示例
array_1d = np.ones(5)  # 创建长度为5的全1数组
array_2d = np.ones((3, 4))  # 创建3行4列的全1二维数组

print(array_1d)
print(array_2d)
print('-----------------zeros()-------------------')
"""np.zeros()是NumPy库中的一个函数，用于创建一个全为0的数组。您可以指定数组的维度和数据类型。
numpy.zeros(shape, dtype=float, order='C')
    shape：指定要创建的数组的形状。
    dtype：指定数组元素的数据类型，默认为float。
    order：指定数组元素在内存中的存储顺序，默认为'C'，即按行存储。

# 创建一个全为0的一维数组
zeros_1d = np.zeros(shape, dtype=type)
# 其中shape是一个整数（例如：5）或者元组（例如：(5,)）

# 创建一个全为0的二维数组
zeros_2d = np.zeros(shape=(rows, cols), dtype=type)
"""
# 示例
array_1d = np.zeros(5)  # 创建长度为5的全0数组
array_2d = np.zeros((3, 4), dtype=float)  # 创建3行4列的全0二维数组

print(array_1d)
print(array_2d)
print('-----------------full()-------------------')
"""np.full()是NumPy库中的一个函数，用于创建一个指定形状并用给定值填充的数组。这个函数接受两个必需参数：shape和fill_value。
numpy.full(shape, fill_value, dtype=None, order='C')

    shape (必需)：是一个整数或元组，用来定义输出数组的维度和大小。例如，如果你想要创建一个3行4列的二维数组，那么shape应为(3, 4)。
    fill_value (必需)：是一个数值或者任何可以被NumPy支持的数据类型所接受的值，这个值会被填入到生成数组的所有元素中。
    dtype (可选)：如果提供，将决定数组元素的数据类型，默认情况下根据fill_value自动推断数据类型。
    order (可选)：内存布局选项， 'C'（默认）表示按行优先存储（C-style）， 'F' 表示按列优先存储（Fortran-style）。
        对于全值填充的数组，此参数通常不会改变结果，因为它不影响元素内容。


# 创建一个指定形状并用特定值填充的一维数组
full_1d = np.full(shape, fill_value, dtype=type) 

# 创建一个指定形状并用特定值填充的二维数组
full_2d = np.full(shape=(rows, cols), fill_valu e=value, dtype=type)
"""
array_1f = np.full(5, fill_value=9)
array_2f = np.full((3, 4), fill_value=1, dtype=float)
print(array_1f)
print(array_2f)

print('-----------------identity()-------------------')
"""np.identity()函数用于生成一个单位矩阵，即对角线元素为1，其余元素为0的方阵。其参数如下：

numpy.identity(n, dtype=None)
    n (必需)：这是一个整数，表示要创建的单位矩阵的维度（行数和列数）。单位矩阵总是方阵。
    dtype (可选)：指定生成矩阵中元素的数据类型，默认是 float64。
        可以根据需要设置为任何NumPy支持的数据类型，如 int, float32, complex 等。
"""
array_i1 = np.identity(5, dtype=float)
print(array_i1)
print(array_i1, type(array_i1))  # 123

print('-----------------一维数组和二维数组的切片是浅复制-------------------')
print('-----------------一维数组的索引-------------------')
"""Python的Numpy库中，一维数组的正索引和负索引如下：
正索引：
    正索引从0开始，表示数组的第一个元素。
    例如，对于一个长度为5的一维数组，索引0表示第一个元素，索引1表示第二个元素，以此类推。
    使用正索引可以方便地访问数组中的特定元素。
负索引：
    负索引从-1开始，表示数组的最后一个元素。
    例如，对于一个长度为5的一维数组，索引-1表示最后一个元素，索引-2表示倒数第二个元素，以此类推。
    使用负索引可以方便地从数组的末尾开始访问元素。
"""
array_1w = np.array([11, 22, 33, 44, 55, 66, 77, 88])
print(array_1w.shape[0])
for i in range(array_1w.shape[0]):
    print(array_1w[i])

ab = array_1w.shape[0]
for i in range(-1, 0 - ab, -1):  # 当需要从大到小遍历时，需要指定负步长
    print(array_1w[i])

"""负索引的元素索引
    0，-9，-8，-7，-6，-5，-4，-3，-2，-1
    正索引的元素索引
    0，1，2，3，4，5，6，7，8，9
"""
# print(array_1w[-9])
print(array_1w[0])
print('-----------------一维数组的切片-------------------')
"""一维数组的切片和list的切片一样，左闭右开"""
print(array_1w[1:3], array_1w[1:3].shape)  # 切出来的是42个元素的一维数组
print(array_1w[1:-1])

print('-----------------二维数组的索引-------------------')
array_2w = np.array([[11, 22, 33, 44, 55],
                     [66, 77, 88, 99, 100],
                     [21, 31, 41, 51, 61],
                     [19, 29, 39, 49, 59],
                     [101, 202, 303, 404, 505]])

a = []
for i in range(array_2w.shape[0]):
    for j in range(array_2w.shape[1]):
        a.append(array_2w[i][j])

print(a)
print(array_2w[-1][-1], array_2w[-3][-3], array_2w[0][0])
print(array_2w[-1, -1], array_2w[-3, -3], array_2w[0, 0])

print('-----------------二维数组的切片-------------------')
bb = array_2w[1:2, 1:2]
print(bb, bb.shape)  # [[77]] (1, 1)  #切出来的是1行1列的二维数组
cc = array_2w[1:4, 1:4]
print(cc, cc.shape)  # 切出来的是3行3列的二维数组
"""[[77 88 99]
 [31 41 51]
 [29 39 49]] (3, 3)"""

print(array_2w[-1, :])
print(array_2w[:, 4])  # 打印正向索引的第5列

""" 
二维数组切片知识点

1、如果二维数组的0轴和1轴都是用的切片，则切出来的都是一个二维数组
2、如果二维数组的0轴或者1轴用的是标量，则切出来的是一个一维数组
3、如果二维数组的0轴和1轴都用的标量，则是取值操作
"""

print('-----------------布尔索引（是深复制）-------------------')
"""NumPy中的布尔索引允许您使用一个布尔数组作为索引，以选择原始数组中与该布尔数组相同形状且对应位置为True的元素。
    这种索引方式非常灵活，可以用于过滤数据、选择满足特定条件的元素或子集等操作。
布尔索引返回的新数组是原数组的副本，与原数组不共享相同的数据空间，即新数组的修改不会影响原数组————深复制
"""
# 一维数组的布尔索引
arr = np.array([4, 2, 9, 6, 3, 7, 1, 5, 8])
condition = arr % 2 == 0  # 创建一个布尔数组，判断是否为偶数
print(condition)
even_numbers = arr[condition]
print(even_numbers, even_numbers.shape)  # 输出：[4 2 6 8] (4,)，即原数组中所有的偶数,一维数组

"""多条件布尔索引： 当需要同时满足多个条件时，可以使用&（逻辑与）、|（逻辑或）和~（逻辑非）运算符来组合条件：
"""
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
condition1 = arr > 4
condition2 = arr < 8
selected_elements = arr[condition1 & condition2]  # ----输出的是一维数组
print(selected_elements, selected_elements.shape)  # 输出：[5 6 7] (3,)，即原数组中大于4且小于8的所有元素
"""直接在索引中使用条件表达式：
"""
arr = np.array([0, 1, -2, 3, -4, 5])
positive_or_zero = arr[arr >= 0]
print(positive_or_zero)  # 输出：[0 1 3 5]，即原数组中非负数的所有元素

# 二维数组的布尔索引
arr2d = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
row_condition = arr2d[:, 1] > 50  # 根据第二列的值选择行,
"""这里 row_condition 是根据第二列生成的一个一维布尔数组，它的长度为3，与 arr2d 的行数相同。
    这意味着 row_condition 中的每个元素对应于 arr2d 的每一行。"""
print(row_condition)  # [False False  True]
selected_rows = arr2d[row_condition, :]  # 切出来的值是一个二维数组
"""NumPy会使用 row_condition 中的布尔值来选择 arr2d 中对应的行。
    当 row_condition 中某个位置的值为 True 时，该行将被选入 selected_rows 中。
    由于 row_condition 的长度与 arr2d 的行数一致，因此这是一种有效的布尔索引操作，
    并且结果 selected_rows 将是一个二维数组，其包含了原始数组中满足条件的行子集
"""
print(selected_rows, '-----', selected_rows.shape)  # 输出：[[70 80 90]]，即原数组中第二列大于50的行
print('----')
lie_condition = arr2d[1, :] >= 60  # 根据第二行中大于或等于60的条件筛选出所有行相同位置的列
selected_lies = arr2d[:, lie_condition]
print(selected_lies, selected_lies.shape)  # 切出来的是一个二维数组

print('-----------------切片索引是浅复制；布尔索引是深复制-------------------')
selected_rows[0, 2] = 99999999
print(selected_rows)
print(arr2d)
print('----------------')
arr2d_ = arr2d[1:, 1:]
print(arr2d_)
arr2d_[1, 1] = 1000000
print(arr2d_)
print(arr2d)

print('----------------花式索引-------------------')
"""
NumPy的花式索引（Fancy indexing）是一种强大的索引机制，它允许你使用整数数组、布尔数组或其他类型的索引器来选择数组中的特定元素或子集。
    这种方式提供了更多的灵活性，特别是在处理高维数组时。

1、花式索引返回的新数组与--花式索引数组--形状相同
2、花式索引返回的新数组与布尔索引类似，属于深层复制
3、二维数组上每一个轴的索引数组相同
"""
# 通过传递一个整数数组作为索引，可以从数组中选择相应位置的元素。这些整数不必连续，也不必从小到大排序。
arr = np.array([33, 11, 32, 83, 94, 52, 16, 107, 28, 39])
fancy_indices = [3, 1, 7, 9]  # 索引列表
arr11 = np.array([3, 1, 7, 9])  # 同样可作为索引
selected_elements = arr[fancy_indices]
print(selected_elements)
print(arr[arr11])

"""段代码使用了NumPy的索引功能，通过arr[d]的方式获取arr数组中特定位置的元素。 
    具体来说，d是一个2x2的数组，它表示了我们要从arr中选取元素的行和列的索引。
例如，d[0,0]表示选取arr的第一行第一列的元素，d[1,1]表示选取arr的第二行第二列的元素。 
因此，arr[d]的意思是根据d中每个元素的行和列索引来选取arr中的元素。
具体来说，选取的元素为：
    arr[d[0,0]] = arr[1] = 11
    arr[d[0,1]] = arr[2] = 32
    arr[d[1,0]] = arr[3] = 83
    arr[d[1,1]] = arr[4] = 94
"""
d = np.array([[1, 2],
              [3, 4]])
print(arr[d])  #
# [[11 32]
# [83 94]]


# 布尔数组索引： 使用布尔数组作为索引时，布尔数组的长度必须与被索引数组的轴长度相同，且索引的位置对应着原数组的元素，True 对应的元素会被选择出来
arr = np.array([10, 20, 30, 40, 50])
bool_mask = arr > 30
selected_elements = arr[bool_mask]
print(selected_elements)
print('-------')

# 多维花式索引： 对于多维数组，花式索引可以同时作用于多个轴。
arr2d = np.array([[10, 20, 30],
                  [40, 50, 60],
                  [70, 80, 90]])

row_indices = [1, 2]
col_indices = [0, 2]
row_indices1 = np.array([1, 2])
col_indices1 = np.array([0, 2])
selected_submatrix = arr2d[row_indices, col_indices]
selected_submatrix1 = arr2d[row_indices1, col_indices1]  # 效果同上，
print(selected_submatrix)  # [40 90]
print('1111')
"""由于row_indices和col_indices都是列表，并且它们的长度相同，NumPy会进行广播（broadcasting），但不是以常规的二维子矩阵方式返回结果。

在NumPy中，当使用两个一维数组作为多维数组的索引时，它将----按照元组对的形式---分别从行和列中--选择--元素。

在这种情况下，arr2d[row_indices, col_indices]实际上是按顺序取每个(row_index, col_index)对的值：
    第一对是 (1, 0)，所以选取的是第二行第一列的元素 40。
    第二对是 (2, 2)，所以选取的是第三行第三列的元素 90。
因此，输出的结果将会是 [40, 90] 而不是一个二维子矩阵。如果想要获取一个二维子矩阵，请确保行和列的索引提供的是完整的子集，

例如使用嵌套列表结构或使用整数切片。正确的子矩阵选择应为
selected_submatrix = arr2d[row_indices[:, np.newaxis], col_indices]
"""
arr_2y = arr2d[row_indices, 1]  # 选取row_indices行的第二列
print(arr_2y)  # [50 80]
print(arr2d[row_indices,])
# [[40 50 60]
#  [70 80 90]]
arr_2y1 = arr2d[row_indices][1]  # arr2d[row_indices] ：第二和第三行的二维数组，arr2d[row_indices][1]:二维数组的第二行
print(arr_2y1)  # [70 80 90]

print('-----')
# 传入二维数组作为二维数组的索引
m = np.array([[1, 1],
              [2, 0]])
n = np.array([[1, 0],
              [2, 2]])

selected_submatrix2 = arr2d[m, n]
print(selected_submatrix2)
# [[50 40]
#  [90 30]]

print('----------------concatenate()、vstack()、hstack()-;拼接多个数组时，数组需要具有相同的维度------------------')
print('----------------concatenate()-------------------')
"""沿指定的轴连接多个数组
用于连接（或拼接）多个数组的函数。它允许沿着指定轴（axis）将一维、二维或多维数组连成一个新的数组。

函数签名： numpy.concatenate(tup, axis=0, out=None)

参数说明：
    tup：一个元组或者列表，其中包含了要进行拼接的数组序列，如 (a1, a2, ..., an)。
        除了沿拼接轴（axis）上的尺寸可以不同之外，这些数组在其他轴的方向上必须具有匹配的形状，
    axis（可选，默认值为0）：整数，表示沿哪个轴进行拼接操作。
        如果 axis=0，则垂直拼接（即按行拼接），增加的是数组的第一维度（通常理解为增加更多的行）。
        如果 axis=1，则水平拼接（即按列拼接），增加的是数组的第二维度（通常理解为增加更多的列）。
    对于多维数组，可以选择其他轴进行拼接。
    out（可选）：如果提供，则输出的数据会直接写入到这个预先分配的数组中。
"""
# 创建两个一维数组
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# 按默认方式（axis=0）拼接
result_1d = np.concatenate((arr1, arr2))
print(result_1d)  # 输出: [1 2 3 4 5 6]

# 创建两个二维数组
arr3 = np.array([[7, 8], [9, 10]])
arr4 = np.array([[11, 12]])

# 按行（axis=0）拼接二维数组
result_2d_rows = np.concatenate((arr3, arr4), axis=0)
print(result_2d_rows)
# 输出: [[ 7  8]
#        [ 9 10]
#        [11 12]]

# 按列（axis=1）拼接二维数组
result_2d_cols = np.concatenate((arr3, arr4.T), axis=1)
print(result_2d_cols)
# 输出: [[ 7  8 11]
#        [ 9 10 12]]
print('----------------vstack()-------------------')
"""沿垂直堆叠多个数组，相当于concatenate()中axis=0的情况，axis=1上的元素个素要相同
numpy.vstack(tup)
"""

# 创建两个一维数组
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# 使用 vstack 函数将它们垂直堆叠起来
result = np.vstack((arr1, arr2))
print(result)
# 输出: [[1, 2, 3],
#        [4, 5, 6]]

# 或者创建两个二维数组
arr3 = np.array([[7, 8], [9, 10]])
arr4 = np.array([[11, 12]])

# 使用 vstack 将它们也垂直堆叠起来
result_2d = np.vstack((arr3, arr4))
print(result_2d)
# 输出: [[ 7,  8],
#        [ 9, 10],
#        [11, 12]]
print('----------------hstack()-------------------')
"""沿水平方向堆叠多个数组，相当于concatenate()中axis=1的情况，axis=0上的元素个素要相同
numpy.Hstack(tup)
"""
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.array([7, 8, 9])
result = np.hstack((a, b, c))
print(result)  # [1 2 3 4 5 6 7 8 9]

d = np.array([[1, 2, 5],
              [3, 4, 7],
              [5, 7, 11]])
e = np.array([[1, 2, 3]])
result_3d = np.hstack((d, e.T))
print(result_3d)

print('----------------split()-------------------')
"""
用于将一个多维数组分割成多个子数组，沿着指定的轴进行分割。这个函数可以按照行（axis=0）或列（axis=1）或其他更高维度的方向来切割原数

numpy.split(ary, indices_or_sections, axis=0)
参数说明：
    ary：要分割的原始多维数组。
    indices_or_sections：数组是左闭右开
        如果是一个整数，表示将数组分割成等份，份数由该值决定。
        如果是一个整数列表或一维数组，则这些整数将作为分割点的位置，数组会在这些位置被分割————切片操作。
    axis（可选，默认为0）：沿哪个轴进行分割操作。0代表按行分割，1代表按列分割（对于二维数组而言），对于更高维度的数组，可以选择其他轴。
示例用法：
"""
# 创建一个一维数组
a1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
aa = np.split(a1, 3)  # [array([1, 2, 3]), array([4, 5, 6]), array([7, 8, 9])]
print(aa)

a2 = np.arange(11, 55, 2)
aaaa = np.split(a2, [7])
print(aaaa)
"""
[array([11, 13, 15, 17, 19, 21, 23]), 
array([25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53])]
"""
aa2 = np.array([4, 7])  # 按照这个索引进行切片，分成了三个数组
aaa = np.split(a2, indices_or_sections=aa2)
print(
    aaa)
""""
[array([11, 13, 15, 17]), 
array([19, 21, 23]), 
array([25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53])]
"""
# 创建一个二维数组
arr = np.array([[11, 13, 15, 17],
                [25, 27, 29, 31],
                [7, 8, 9, 10],
                [37, 39, 41, 43]])

# 按行分割，将数组分成4部分
split_arrs = np.split(arr, 4, axis=0)
print(
    split_arrs)  # [array([[11, 13, 15, 17]]), array([[25, 27, 29, 31]]), array([[ 7,  8,  9, 10]]), array([[37, 39, 41, 43]])]

split_arrs1 = np.split(arr, indices_or_sections=[1, 2], axis=0)  # 按照行索引（第二个素组）进行切片
print(split_arrs1)  # [array([[11, 13, 15, 17]]), array([[25, 27, 29, 31]]), array([[ 7,  8,  9, 10],[37, 39, 41, 43]])]

split_arrs2 = np.split(arr, indices_or_sections=[1, 2], axis=1)  # 按照列索引（第二个素组）进行切片
print(split_arrs2)
"""
[array([[11],
       [25],
       [ 7],
       [37]]),
array([[13],
       [27],
       [ 8],
       [39]]), 
array([[15, 17],
       [29, 31],
       [ 9, 10],
       [41, 43]])]
"""
print('----------------vsplit()-------------------')
"""专门用于垂直分割多维数组，即沿着数组的第一个轴（axis=0）进行分割。
该函数接收一个数组和一个分割点列表或一个整数作为参数，将原始数组拆分为多个子数组

numpy.vsplit(ary, indices_or_sections)

参数说明：
    ary：要分割的多维数组。
    indices_or_sections：
    如果是一个整数 n，则将数组平均分割成 n 个子数组（如果无法平均分配，则最后一块可能会较小）。
    如果是一个整数列表或一维数组，则数组将在这些指定的索引位置进行分割。
"""
# 创建一个二维数组
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
                [25, 27, 29],
                [39, 41, 43]])

split_arrs11 = np.split(arr, indices_or_sections=[4])  # 从第4行开始分割，分割成了两部分，第5行在第二部分中
print(split_arrs11)
"""
[array([[ 1,  2,  3],
       [ 4,  5,  6],
       [ 7,  8,  9],
       [10, 11, 12]]),
array([[25, 27, 29],
       [39, 41, 43]])]
"""
# 垂直分割，在第二行和第四行进行分割，左闭右开
split_arrs111 = np.vsplit(arr, [1, 3])
print(split_arrs111)
"""
[array([[1, 2, 3]]), 
array([[4, 5, 6],[7, 8, 9]]), 
array([[10, 11, 12],
       [25, 27, 29],
       [39, 41, 43]])]
"""
# 若传入一个整数，将数组平均分割成两部分
split_arrs_half = np.vsplit(arr, 2)
print(split_arrs_half)
"""
[array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]]), 
array([[10, 11, 12],
       [25, 27, 29],
       [39, 41, 43]])]
"""
print('----------------hsplit()-------------------')
"""用于水平分割多维数组，即沿着数组的第二个轴（axis=1）进行分割。
这个函数接收一个数组和一个分割点列表或一个整数作为参数，将原始数组沿着水平方向拆分成多个子数组。

函数签名：numpy.hsplit(ary, indices_or_sections)

ary：要分割的多维数组。
    indices_or_sections：
    如果是一个整数 n，则将数组的每一行在列方向上平均分割成 n 个子数组（如果无法平均分配，则最后一块可能会较小）。
    如果是一个整数列表或一维数组，则数组的每一行将在这些指定的列索引位置进行分割。
"""

# 创建一个二维数组
arr = np.array([[1, 2, 3, 4, 11, 22, 33, 44], [5, 6, 7, 8, 111, 222, 333, 444]])

# 水平分割，将数组在第二个元素处分割
split_arrs = np.hsplit(arr, [2])  # 从第三列分割，分割成了两部分，第三列在第二部分中
print(split_arrs)
"""
[array([[1, 2],[5, 6]]), 
array([[  3,   4,  11,  22,  33,  44],
       [  7,   8, 111, 222, 333, 444]])]
"""
split_arrs1 = np.hsplit(arr, [2, 3])
print(split_arrs1)
"""
[array([[1, 2], [5, 6]]), 
array([[3],[7]]), 
array([[  4,  11,  22,  33,  44],
       [  8, 111, 222, 333, 444]])]
"""
# 若传入一个整数，将数组的每一行平均分割成两部分
split_arrs_half = np.hsplit(arr, 2)
print(split_arrs_half)
"""
[array([[1, 2, 3, 4], 
        [5, 6, 7, 8]]), 
array([[ 11,  22,  33,  44],
       [111, 222, 333, 444]])]
"""
print('----------------数组运算-------------------')
"""
数学运算：
算术运算：+、-、*、/、//（整数除）、**（乘方）

数学函数：如 numpy.sin()、numpy.cos()、numpy.tan()、numpy.sqrt()（平方根）、numpy.log()（自然对数）、numpy.exp()（指数函数）

矩阵运算：
    矩阵乘法：使用 numpy.dot() 或 @ 运算符进行点积，对于二维数组，numpy.matmul() 可用于通用矩阵乘法。
    矩阵转置：numpy.transpose() 或属性 .T
    内积与外积：numpy.inner() 和 numpy.outer()
数组操作：
    形状变换：numpy.reshape()、numpy.resize()、numpy.ravel()（展平为一维数组）
    元素级操作：如 numpy.maximum()、numpy.minimum()（元素级最大值、最小值）、numpy.clip()（限制元素范围）
    广播机制：不同形状的数组在符合广播规则的情况下能自动匹配并进行运算
合并与拆分数组：
    合并：numpy.concatenate()（连接）、numpy.vstack()（垂直堆叠）、numpy.hstack()（水平堆叠）
    拆分：numpy.split()、numpy.vsplit()（垂直分割）、numpy.hsplit()（水平分割）
统计和汇总运算：
    统计函数：numpy.mean()（平均值）、numpy.median()（中位数）、numpy.std()（标准差）、numpy.var()（方差）
    函数应用于数组所有元素：如 numpy.sum()（求和）、numpy.min()（最小值）、numpy.max()（最大值）
索引和切片：
    使用索引访问或修改数组元素：arr[index]
    切片获取子数组：arr[start:stop:step]
    高级索引：使用布尔型数组、列表或其他数组作为索引
条件逻辑：
    逻辑运算：numpy.logical_and()、numpy.logical_or()、numpy.logical_not()
    根据条件选择：numpy.where(condition, x, y) 返回满足条件的 x 值，否则返回 y 值
类型转换：
    类型转换函数：numpy.astype() 将数组转换为指定数据类型
排序：
    排序函数：numpy.sort() 对数组进行就地排序，numpy.argsort() 返回排序后的索引
"""
print('----------------一维数组算数运算-------------------')
