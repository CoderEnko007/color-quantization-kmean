#python quant.py --image images\test1.jpg --cluster 8
from sklearn.cluster import MiniBatchKMeans
import imutils
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True)
ap.add_argument("-c", "--clusters", type = int, required = True)
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
image = cv2.bilateralFilter(image, d = 9, sigmaColor = 9, sigmaSpace = 7)
cv2.imshow("blur", imutils.resize(image, height = 500))
image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
image = cv2.medianBlur(image, 7)
image = image.reshape(image.shape[0] * image.shape[1], 3)

'''
>>> from sklearn.cluster import MiniBatchKMeans
>>> import numpy as np
>>> X = np.array([[1,2], [1,4], [1,0], [4,2], [4,4], [4,0]])
>>> clt = MiniBatchKMeans(n_clusters = 2)
>>> label = clt.fit_predict(X)
>>> label
array([0, 0, 0, 1, 1, 1])
>>> clt.cluster_centers_
array([[ 1.        ,  2.03626943],
       [ 4.        ,  2.06811989]])
>>> clt.cluster_centers_.astype("uint8")
array([[1, 2],
       [4, 2]], dtype=uint8)
>>> clt.cluster_centers_.astype("uint8")[label]
array([[1, 2],
       [1, 2],
       [1, 2],
       [4, 2],
       [4, 2],
       [4, 2]], dtype=uint8)
'''
clt = MiniBatchKMeans(n_clusters = args["clusters"])
label = clt.fit_predict(image)
quant = clt.cluster_centers_.astype("uint8")[label]

image = image.reshape(h, w, 3)
quant = quant.reshape(h, w, 3)

image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)

image = imutils.resize(image, height = 500)
quant = imutils.resize(quant, height = 500)

cv2.imshow("image", np.hstack([image, quant]))
cv2.waitKey(0)