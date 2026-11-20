import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from jacobi_package import jacobi_eigen

def test_diagonal_matrix():
    """тест для уже диагональной матрицы."""
    A = np.diag([1.0, 2.0, 3.0])
    vals, vecs = jacobi_eigen(A)
    np.testing.assert_allclose(np.sort(vals), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(np.abs(vecs), np.eye(3), atol=1e-7)

def test_simple_matrix():
    """тест для матрицы 2*2."""
    A = np.array([[2.0, 1.0],
                  [1.0, 2.0]])
    vals, vecs = jacobi_eigen(A)

    expected_vals = np.array([1.0, 3.0])
    np.testing.assert_allclose(np.sort(vals), expected_vals, rtol=1e-7)

    for i in range(2):
        v = vecs[:, i]
        lam = vals[i]
        np.testing.assert_allclose(A @ v, lam * v, atol=1e-7)

def test_random_symmetric_matrix():
    """тест на случайной симметричной матрице большего размера."""
    np.random.seed(42)
    n = 5
    M = np.random.rand(n, n)
    A = (M + M.T) / 2

    vals, vecs = jacobi_eigen(A)

    np.testing.assert_allclose(vecs.T @ vecs, np.eye(n), atol=1e-7)

    for i in range(n):
        v = vecs[:, i]
        lam = vals[i]
        np.testing.assert_allclose(A @ v, lam * v, atol=1e-7)

def test_error_non_square():
    """проверка, что не квадратная матрица вызывает ошибку."""
    A = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="матрица должна быть квадратной"):
        jacobi_eigen(A)

def test_error_non_symmetric():
    """проверка, что несимметричная матрица вызывает ошибку."""
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="матрица должна быть симметричной"):
        jacobi_eigen(A)

def test_zero_matrix():
    """тест для нулевой матрицы."""
    A = np.zeros((3, 3))
    vals, vecs = jacobi_eigen(A)
    np.testing.assert_allclose(vals, np.zeros(3))
    np.testing.assert_allclose(vecs.T @ vecs, np.eye(3), atol=1e-7)