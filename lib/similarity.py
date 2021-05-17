import numpy as np


def cos(A, B, norm=True):
    num = A.T.dot(B)  # 若为行向量则 A * B.T
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    _cos = num / denom
    if norm: _cos = 0.5 + 0.5 * _cos
    return _cos


if __name__ == '__main__':
    A = np.array([1, 1, 1, 0])
    B = np.array([1, 1, 0, 1])
    sim = cos(A, B)
    print(sim)
