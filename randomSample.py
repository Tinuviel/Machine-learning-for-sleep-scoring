matrixRowSize = matrixA.shape[1]
A = matrixA
B = matrixB
A = A.T
B = B.T
train_len = 4000
x_train = A[0:train_len,:]
y_train = B[0:train_len,:]
x_test = A[train_len:matrixRowSize,:]
y_test = B[train_len:matrixRowSize,:]

index = [0] * matrixRowSize
for i in range(0, matrixRowSize):
    index[i] = i


for row in range(0, train_len):
    test = np.random.randint(0, matrixRowSize-1)
    x_train[row, :] = A[index[test], :]
    y_train[row, :] = B[index[test], :]
    index.remove(index[test])
    matrixRowSize = matrixRowSize-1

for i in range(0, len(index)):
    x_test[i, :] = A[index[i], :]
    y_test[i, :] = B[index[i], :]
