import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img_eg = mpimg.imread("./input/3.jpg")
# img_eg = img_eg[:, :, :3]
shape = img_eg.shape
print(shape)

img_temp = img_eg.reshape(shape[0], shape[1] * shape[2])
U, Sigma, VT = np.linalg.svd(img_temp)
print(Sigma.shape)
print(Sigma)
# 取前60个奇异值
sval_nums1 = 200
img_restruct1 = (U[:, 0:sval_nums1]).dot(np.diag(Sigma[0:sval_nums1])).dot(VT[0:sval_nums1, :])
img_restruct1 = img_restruct1.reshape(shape[0], shape[1], shape[2])
# img_restruct1[img_restruct1 < 0] = 0
# img_restruct1[img_restruct1 > 1] = 1

# 取前120个奇异值
sval_nums2 = 400
img_restruct2 = (U[:, 0:sval_nums2]).dot(np.diag(Sigma[0:sval_nums2])).dot(VT[0:sval_nums2, :])
img_restruct2 = img_restruct2.reshape(shape[0], shape[1], shape[2])
# img_restruct2[img_restruct2 < 0] = 0
# img_restruct2[img_restruct2 > 1] = 1

fig, ax = plt.subplots(1, 3, figsize=(2400, 3200))

ax[0].imshow(img_eg)
ax[0].set(title="src")
# ax[1].imshow(img_restruct1)
ax[1].imshow(img_restruct1.astype(np.uint8))
ax[1].set(title="nums of sigma = {}".format(sval_nums1))
# ax[2].imshow(img_restruct2)
ax[2].imshow(img_restruct2.astype(np.uint8))
ax[2].set(title="nums of sigma = {}".format(sval_nums2))
plt.show()

# plt.figure(figsize=(6,4))
plt.plot(Sigma, color="red", linewidth=1)
plt.show()
