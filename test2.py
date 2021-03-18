import json
# with open('./output/text_info.json', encoding='utf8') as f:
#     text_info=json.load(f)
#
# print(json.dumps(text_info[14656:14671+1]))

import numpy as np

A = np.array([1, 1, 1, 0])
B = np.array([1, 1, 0,1])
# print(A.T)
num = A.T.dot(B)  # 若为行向量则 A * B.T
denom = np.linalg.norm(A) * np.linalg.norm(B)
cos = num / denom  # 余弦值
# print(cos)
sim = 0.5 + 0.5 * cos  # 归一化
print(sim)
