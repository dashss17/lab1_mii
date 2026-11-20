import numpy as np


def jacobi_eigen(A: np.ndarray, eps: float = 1e-9, max_iter: int = 10000):
    """
    вычисление собственных чисел и векторов действительной симметричной матрицы методом якоби.

    параметры

    A : np.ndarray
        действительная симметричная матрица (n x n).
    eps : float, optional
        точность вычисления (порог для суммы квадратов недиагональных элементов). по умолчанию 1e-9.
    max_iter : int, optional
        максимальное количество итераций для предотвращения бесконечного цикла. по умолчанию 10000.

    возвращает

    eigenvalues : np.ndarray
        одномерный массив собственных чисел (диагональ матрицы).
    eigenvectors : np.ndarray
        матрица, столбцы которой являются собственными векторами.

    исключения

    ValueError
        если входная матрица не является квадратной или симметричной.
    RuntimeError
        если алгоритм не сошелся за заданное число итераций.
    """
    if not isinstance(A, np.ndarray):
        A = np.array(A, dtype=float)
    else:
        A = A.astype(float)

    n, m = A.shape
    if n != m:
        raise ValueError("матрица должна быть квадратной.")

    if not np.allclose(A, A.T, atol=1e-8):
        raise ValueError("матрица должна быть симметричной.")

    A_curr = A.copy()
    V = np.eye(n)
    iterations = 0

    while True:
        if iterations >= max_iter:
            raise RuntimeError(f"алгоритм не сошелся за {max_iter} итераций.")

        # Надежный поиск максимального недиагонального элемента
        max_val = 0.0
        p, q = -1, -1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A_curr[i, j]) > max_val:
                    max_val = abs(A_curr[i, j])
                    p, q = i, j

        # Если максимальный элемент меньше eps, значит сошлись
        if max_val < eps:
            break

        # Вычисление коэффициента c
        C = (A_curr[q, q] - A_curr[p, p]) / (2.0 * A_curr[p, q])

        # Вычисление тангенса
        if C == 0.0:
            t = 1.0
        elif C > 0:
            t = 1.0 / (C + np.sqrt(C ** 2 + 1))
        else:
            t = 1.0 / (C - np.sqrt(C ** 2 + 1))

        c = 1.0 / np.sqrt(1 + t ** 2)
        s = t * c

        # Сохранение старых значений
        a_pp = A_curr[p, p]
        a_qq = A_curr[q, q]
        a_pq = A_curr[p, q]

        # Обновление диагонали
        A_curr[p, p] = a_pp - t * a_pq
        A_curr[q, q] = a_qq + t * a_pq

        # Обнуление
        A_curr[p, q] = 0.0
        A_curr[q, p] = 0.0

        # Обновление остальных элементов
        for r in range(n):
            if r != p and r != q:
                a_rp = A_curr[r, p]
                a_rq = A_curr[r, q]
                k = s / (1.0 + c)

                new_A_rp = a_rp - s * (a_rq + k * a_rp)
                new_A_rq = a_rq + s * (a_rp - k * a_rq)

                A_curr[r, p] = new_A_rp
                A_curr[p, r] = new_A_rp
                A_curr[r, q] = new_A_rq
                A_curr[q, r] = new_A_rq

        # Обновление собственных векторов
        for r in range(n):
            v_rp = V[r, p]
            v_rq = V[r, q]
            V[r, p] = v_rp * c - v_rq * s
            V[r, q] = v_rp * s + v_rq * c

        iterations += 1

    eigenvalues = np.diag(A_curr)
    return eigenvalues, V