from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression

import numpy as np 
import argparse
import imutils
import time
import cv2

def decode_pred(scores,geo):
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
    return (rects,conf)

ap = argparse.ArgumentParser()
ap.add_argument("-east","--east",type = str,required = True)
ap.add_argument("-v","--video",type = str)
ap.add_argument("-c","--min-confidence",type = float,default = 0.5)
ap.add_argument("-w","--width",type = int,default =320)
ap.add_argument("-e","--height",type = int,default = 320)

args = vars(ap.parse_args())


(W,H) = (None,None)
(newW,newH) = (args["width"],args["height"])
(rW,rH) = (None,None)




layerN  = ["feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]


print("loading....\n")
net = cv2.dnn.readNet(args["east"])


if not args.get("video",False):
    print("Starting vid stream")
    vs = VideoStream(arc = 0).start()
    time.sleep(1.0)

else:
    vs = cv2.VideoCapture(args["Video"])

fps = FPS().start()


while True:

    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame

    if frame is None:
        break

    frame  = imutils.resize(frame,width = 1000)
    im2 = frame.copy()



    if W is None or H is None:
        (H,W) = frame.shape[:2]
        rW = W/float(newW)
        rH = H/float(newH)

    frame = cv2.resize(frame,(newW,newH))


    blob = cv2.dnn.blobFromImage(frame,1.0,(newW,newH),
        (123.68,116.78,103.94),swapRB = True,crop = False)

    net.setInput(blob)
    (scores, geo) = net.forward(layerN)

    (rects,conf) = decode_pred(scores,geo)
    boxes = non_max_suppression(np.array(rects),probs = conf)

    for (startX,startY,endX,endY) in boxes:

        startX = int(startX*rW)
        startY = int(startY*rH)
        endX = int(endX*rW)
        endY = int(endY*rH)


        cv2.rectangle(im2,(startX,startY),(endX,endY),(0,255,0),2)

    fps.update()

    cv2.imshow("Vid",im2)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

fps.stop()
print("Elapsed time:{:.2f}".format(fps.elapsed()))
print("Approx fps: {:.2f}".format(fps.fps()))

if not args.get("video",False):
    vs.stop()

else:
    vs.release()

cv2.destroyAllWindows()