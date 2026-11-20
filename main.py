import numpy as np
from jacobi_package import jacobi_eigen

# Тестовая матрица 2x2
A = np.array([[2.0, 1.0],
              [1.0, 2.0]])

eigenvalues, eigenvectors = jacobi_eigen(A)

print("Собственные числа:")
print(eigenvalues)
print("\nСобственные векторы (по столбцам):")
print(eigenvectors)