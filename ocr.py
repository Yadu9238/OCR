from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",type = str)
ap.add_argument("-east","--east",type = str)
ap.add_argument("-c","--min-confidence",type = float,default = 0.5)
ap.add_argument("-w","--width",type = int,default = 320)
ap.add_argument("-e","--height",type = int,default = 320)

args = vars(ap.parse_args())




img = cv2.imread(args["image"])
im2 = img.copy()
(H,W) = img.shape[:2]

(newW,newH) = (args["width"],args["height"])
rW = W/float(newW)
rH = H/float(newH)

img = cv2.resize(img,(newW,newH))
(H, W) = img.shape[:2]




layerN = ["feature_fusion/Conv_7/Sigmoid",
          "feature_fusion/concat_3"]


print("Loading...\n")
net = cv2.dnn.readNet(args["east"])

blob = cv2.dnn.blobFromImage(img,1.0,(W,H),(123.68,116.78,103.94),swapRB = True, crop = False)
start = time.time()
net.setInput(blob)
(scores,geo) = net.forward(layerN)
end = time.time()

print("Time took: {:.6f} ".format(end-start))

(numR,numC) = scores.shape[2:4]
rects = []
conf = []

for y in range(0,numR):
    scoresD = scores[0,0,y]
    xdata0 = geo[0,0,y]
    xdata1 = geo[0,2,y]
    xdata2 = geo[0,2,y]
    xdata3 = geo[0,3,y]
    angle = geo[0,4,y]


    for x in range(0,numC):
        if scoresD[x] < args["min_confidence"]:
            continue

        (offsetX,offsetY) = (x*4.0,y*4.0)


        angleD = angle[x]
        cos = np.cos(angleD)
        sin = np.sin(angleD)

        h = xdata0[x] + xdata2[x]
        w = xdata1[x] + xdata3[x]


        endX = int(offsetX + (cos*xdata1[x])+(sin*xdata2[x]))
        endY = int(offsetY - (sin*xdata1[x]) + (cos*xdata2[x]))
        startX = int(endX -w)
        startY = int(endY - h)
        rects.append((startX,startY,endX,endY))
        conf.append(scoresD[x])

boxes = non_max_suppression(np.array(rects),probs = conf)

for (startX,startY,endX,endY) in boxes:

    startX = int(startX*rW)
    startY = int(startY*rH)
    endX = int(endX*rW)
    endY = int(endY*rH)

    cv2.rectangle(im2,(startX,startY),(endX,endY),(0,255,0),2)

cv2.imshow("Text Detected",im2)
cv2.imwrite('output/output2.jpg',im2)
cv2.waitKey(0)
