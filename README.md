# Optical character Detection & Recognition

## Introduction
OCR is the method to detect text from images or live cam and convert it to machine-encoded text.I have used 
OpenCV's EAST(Efficient and Accurate Scene Text Detection), which is a deep learning model created to obtain 
high accuracy in text detection

## Requirements

You need:
- **OpenCV**
- **Python**
- **numpy**
- **EAST model**: [Link to download](https://www.dropbox.com/s/dl/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz)

## Run the code

- Download and place the .pb file as the same folder as this
- The options provided are:
'''
usage: ocr.py [-h] [-i IMAGE] [-east EAST] [-c MIN_CONFIDENCE] [-w WIDTH] [-e HEIGHT]

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE, --image IMAGE
  -east EAST, --east EAST
  -c MIN_CONFIDENCE, --min-confidence MIN_CONFIDENCE
  -w WIDTH, --width WIDTH
  -e HEIGHT, --height HEIGHT

'''

## limitations
As of now, it does detect text but not completely.As seen in the below examples
![](output/output1.jpg) ![] (output/output2.jpg)

## To-do
Implement character-recognition from the detected text,using Tessaract or other OCR engine
