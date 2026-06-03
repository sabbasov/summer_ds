import numpy as np

# create a 2x2 matrix
matrix1 = np.array([[1, 3], 
                   [5, 7]])

print("2x2 Matrix:\n",matrix1)

# create a 3x3  matrix
matrix2 = np.array([[2, 3, 5],
             	    [7, 14, 21],
                    [1, 3, 5]])
                    
print("\n3x3 Matrix:\n",matrix2)

matrix3 = np.array([[1, 3],
                   [5, 7]])

matrix4 = np.array([[2, 6],
                   [4, 8]])

result = np.dot(matrix3, matrix4)

print("\nmatrix3 x matrix4: \n", result)

transposed_result = np.transpose(result)

print("\ntransposed result: \n", transposed_result)

inversed_result = np.linalg.inv(result)

print("\n inversed result: \n", inversed_result)

dot_result_inverse = np.dot(result, inversed_result)

print("\n dot product of result and its inverse: \n", dot_result_inverse)

det_result = np.linalg.det(result)

print("\n determinant of result: \n", det_result)

result_flat = result.flatten()
print("\n flattened result: \n", result_flat)

eigenvalues, eigenvectors = np.linalg.eig(result)

print(f"\n eigenvalues: {eigenvalues}\n")
print(f"\n eigenvalues: {eigenvalues}\n")

a = 5
b = a
c = 5

if 5 == b:
    print("5 is b")