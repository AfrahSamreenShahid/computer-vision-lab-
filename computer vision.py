# to import image on to the python using opencv
import cv2 as cv
import numpy
image_black=np.zeros((256,256))
image_1=np.ones((256,256,3))*255
cv.imshow("black", image_black)
cv.waitKey(0)


kf_image=cv.imread(r"Kingfisher.jpg")
cv.imshow("image", kf_image)
cv.waitKey(0)

import matplotlib.pyplot as plt
plt.imshow(kf_image)

# lets load another images 
mush=cv.imread(r"C:\Users\afrah\Downloads\mush.jpg")
cv.imshow("mush", mush)
cv.waitKey(0)
# crop1=kf_image[0:250,0:250,0]
# image_black[0:250,0:250]=crop1
# black_tyre=tyre[100:,100:,0]
# tyre[100:,100:,0]=image_1[:,:,0]
# black_tyre[100:356,100:356]=image_1[:,:,0]
#the size of the image is too big lets reduce it 
dim=(1024,786)
resized_tyre = cv.resize(tyre,dim, interpolation = cv.INTER_AREA)
cv.imshow("tyred", resized_tyre)
cv.waitKey(0)

# their are few unwanted object in images lets crop them 
tyredup=resized_tyre
crop_img = tyredup[250:586, 400:824]
cv.imshow("tyre", crop_img)
cv.waitKey(0)
# Saving an Image to Disk using Python and OpenCV
cv.imwrite("crop_img.png", crop_img)
#Image Blurring This is done by convolving the image with a normalized box filter. 
#It simply takes the average of all the pixels under kernel area and replaces the central element with this average.
blur = cv.blur(resized_tyre,(3,3))
blur = cv.blur(resized_tyre,(7,7))
cv.imshow("tyre", blur)
cv.waitKey(0)
# Image Thresholding
img = cv.imread('crop_img.png',0)
#Global Thresholding
img = cv.imread('C:\Users\afrah\Downloads\mush.jpg',0)
ret,th1 = cv.threshold(img,125,255,cv.THRESH_BINARY)
cv.imshow("mush", th1)
cv.waitKey(0)

#AVG and gaussian thersholding 
img = cv.imread('C:\\Users\\afrah\\Downloads\\mush.jpg',0)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
                           cv.THRESH_BINARY,11,2)
cv.imshow("mush", th2)
cv.waitKey(0)

img = cv.imread('C:\\Users\\afrah\\Downloads\\mush.jpg',0)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)
cv.imshow("mush", th3)
cv.waitKey(0)

#Image Gradients
laplacian = cv.Laplacian(img,cv.CV_64F)
#used for vertical and horizontal highlights of edges
sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)
cv.imshow("tyre", sobely)
cv.waitKey(0)
#Canny Edge Detection in OpenCV
edges = cv.Canny(crop_img,50,150)
cv.imshow("tyre", edges)
cv.waitKey(0)
#PLT
edges_50 = cv.Canny(crop_img,50,100)
edges_25 = cv.Canny(crop_img,25,100)
plt.imshow(edges_50,cmap='gray')
cv.imshow("tyre", edges_50)
cv.waitKey(0)
import matplotlib as plt
#Morphological Transformations
#1) Erosion
kernel = np.ones((5,5),np.uint8)
erosion = cv.erode(resized_tyre,kernel,iterations = 1)
cv.imshow("tyre", erosion)
cv.waitKey(0)
# Contours
imges = cv.imread('crop_img.png') #reads img
imgray = cv.cvtColor(imges,cv.COLOR_BGR2GRAY) #changing clr to bgr to grayscale image
gray = cv.bilateralFilter(imgray, 11, 17, 17) #applying bilateral filter(it smoothens the pixel values) to make image sharpen
edged = cv.Canny(gray, 20, 90) 
cv.imshow("tyre", edged)
cv.waitKey(0)

contours, hierarchy = cv.findContours(edged, cv.RETR_TREE, 
                                             cv.CHAIN_APPROX_SIMPLE) #making connections from one to another edge
x,y,w,h = cv.boundingRect(contours[0]) #make rectangle box

for i in range(len(contours)):   #creating a loop to  create contour & create rect and save the img 
    area = cv.contourArea(contours[i])
    x,y,w,h = cv.boundingRect(contours[i])
    imgrect = cv.rectangle(imges,(x,y),(x+w,y+h),(0,255,0),2)
    outfile = ('%s.jpg' % i)
    cv.imwrite(outfile, imgrect)
cv.imshow("tyre", imgrect)
cv.waitKey(0)
cv.destroyAllWindows()

#to draw circles 
(x1,y1),rad = cv.minEnclosingCircle(contours[70])
cen = (int(x1),int(y1))
radi = int(rad)
imgcir = cv.circle(crop_img,cen,radi,(0,255,0),2)
cv.imshow("tyre", imgcir)
cv.waitKey(0)  

# to draw a lot of circles 
for i in range(5):
    (xc,yc),radius = cv.minEnclosingCircle(contours[i])
    center = (int(xc),int(yc))
    radius = int(radius)
    imgcircle = cv.circle(crop_img,center,radius,(0,255,0),2)
cv.imshow("tyre", imgcircle)
cv.waitKey(0)    

#histogram 
from matplotlib import pyplot as plt
kingimg = cv.imread('Kingfisher.png')
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv.calcHist([kingimg],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()

# Harris Corner Detector
colimage=cv.imread('tallestbuilding.png')
graytall = cv.cvtColor(colimage,cv.COLOR_BGR2GRAY)
dst = cv.cornerHarris(graytall,2,3,0.04)
dst = cv.dilate(dst,None)
max=0.001*dst.max()
colimage[dst>max]=[0,0,255]
cv.imshow('dst',colimage)
cv.waitKey(0)

#Brute-Force matcher 
img1 = cv.imread('tyre.jpg',0)          # queryImage
img2 = cv.imread('crop_img.png',0) # trainImage

# Initiate SIFT detector
orb = cv.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 5 matches.
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:25],None,flags=2)
plt.imshow(img3),plt.show()


## program 9 colur fundamentals

import cv2
flags = [i for i in dir(cv) if i.startswith('COLOR_')]

len(flags)
flags[40]

import matplotlib.pyplot as plt
import numpy as np
nemo = cv.imread("C:\Users\afrah\Downloads\nemo.jpg")

plt.imshow(nemo)
plt.show()

nemo = cv.cvtColor(nemo, cv.COLOR_BGR2RGB)
plt.imshow(nemo)
plt.show()

#Visualizing Nemo in RGB Color Space
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
r, g, b = cv.split(nemo)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

pixel_colors = nemo.reshape((np.shape(nemo)[0]*np.shape(nemo)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
plt.show()

# Visualizing Nemo in HSV Color Space

hsv_nemo = cv.cvtColor(nemo, cv2.COLOR_RGB2HSV)
h, s, v = cv.split(hsv_nemo)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()

# picking out range
light_orange = (1, 190, 200)
dark_orange = (18, 255, 255)

mask = cv.inRange(hsv_nemo, light_orange, dark_orange)

result = cv.bitwise_and(nemo, nemo, mask=mask)

plt.subplot(1, 2, 1)
plt.imshow(mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.show()

from matplotlib.colors import hsv_to_rgb
light_white = (0, 0, 200)
dark_white = (145, 60, 255)
lw_square = np.full((10, 10, 3), light_white, dtype=np.uint8) / 255.0
dw_square = np.full((10, 10, 3), dark_white, dtype=np.uint8) / 255.0
plt.subplot(1, 2, 1)
plt.imshow(hsv_to_rgb(lw_square))
plt.subplot(1, 2, 2)
plt.imshow(hsv_to_rgb(dw_square))
plt.show()

mask_white = cv.inRange(hsv_nemo, light_white, dark_white)
result_white = cv.bitwise_and(nemo, nemo, mask=mask_white)
plt.subplot(1, 2, 1)
plt.imshow(mask_white, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result_white)
plt.show()

final_mask = mask + mask_white

final_result = cv.bitwise_and(nemo, nemo, mask=final_mask)
plt.subplot(1, 2, 1)
plt.imshow(final_mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(final_result)
plt.show()


#10 program clusterting and classification
# import the necessary packages
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import utils
import cv2
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-c", "--clusters", required = True, type = int, help = "# of clusters")
args = vars(ap.parse_args())
# load the image and convert it from BGR to RGB so that
# we can dispaly it with matplotlib
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# show our image
plt.figure()
plt.axis("off")
plt.imshow(image)

#11 program dim reduction and sparse representation
#importing the dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
 
digits = load_digits()
data = digits.data
data.shape
#PART1
#taking a sample image to view
#Remember image is in the form of numpy array.
image_sample = data[0,:].reshape(8,8)
plt.imshow(image_sample)


#Import required modules
from sklearn.decomposition import PCA
 
pca = PCA(2) # we need 2 principal components.
converted_data = pca.fit_transform(digits.data)
 
converted_data.shape

plt.style.use('seaborn-whitegrid')
plt.figure(figsize = (10,6))
c_map = plt.cm.get_cmap('jet', 10)
plt.scatter(converted_data[:, 0], converted_data[:, 1], s = 15,
            cmap = c_map , c = digits.target)
plt.colorbar()
plt.xlabel('PC-1') , plt.ylabel('PC-2')
plt.show()
#PART 2
# Importing required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# Loading the image 
img = cv2.imread("C:\\Users\\afrah\\OneDrive\\Desktop\\CV Lab\\sample file.jpg") #you can use any image you want.
plt.imshow(img)


#classification and clustering 
#clustering
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('C:\\Users\\afrah\\OneDrive\\Desktop\\CV Lab\\sample file.jpg')

# Reshape the image data to be a list of RGB pixels
pixels = image.reshape(-1, 3)

# Perform K-Means clustering
kmeans = KMeans(n_clusters=5)  # You can choose the number of clusters
kmeans.fit(pixels)

# Get cluster assignments for each pixel
labels = kmeans.labels_

# Reshape the labels back to the original image shape
segmented_image = labels.reshape(image.shape[:2])

# Display the segmented image using Matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(segmented_image, cmap='viridis')  # You can choose a different colormap
plt.title('Segmented Image')
plt.axis('off')
plt.show()
#classification
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load and preprocess your dataset of labeled images
# This dataset should contain images and corresponding labels
# You need to define 'images' and 'labels' appropriately in your code

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Initialize the k-NN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier on the training data
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Evaluate the classifier using accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Visualize the predicted labels
plt.figure(figsize=(10, 5))
plt.scatter(range(len(y_test)), y_test, label='Actual Labels', marker='o')
plt.scatter(range(len(y_pred)), y_pred, label='Predicted Labels', marker='x')
plt.xlabel('Sample Index')
plt.ylabel('Labels')
plt.title('Actual vs. Predicted Labels')
plt.legend()
plt.show()
