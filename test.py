import cv2
import numpy as np
import operator
import os
import random


MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30


class ContourWithData():

    npaContour = None
    boundingRect = None
    intRectX = 0
    intRectY = 0
    intRectWidth = 0
    intRectHeight = 0
    fltArea = 0.0

    def calculateRectTopLeftPointAndWidthAndHeight(self):
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):
        if self.fltArea < MIN_CONTOUR_AREA: return False
        return True


def sp_noise(image,prob):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def main():
    allContoursWithData = []
    validContoursWithData = []

    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)
    except:
        print ("error, unable to open classifications.txt, exiting program\n")
        os.system("pause")
        return
    # end try

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
    except:
        print ("error, unable to open flattened_images.txt, exiting program\n")
        os.system("pause")
        return
    # end try

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))

    kNearest = cv2.ml.KNearest_create()

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    imgTestingNumbers = cv2.imread("test5.png")
    noise_img = sp_noise(imgTestingNumbers, 0.0)

   # for number in range(0, 500, 3):
    #    noise_img = cv2.line(noise_img, (number, 0), (number, 500), (0,0,0), 1)
    #for number in range(0, 500, 3):
    #    noise_img = cv2.line(noise_img, (0, number), (500, number), (0,0,0), 1)

    if  noise_img is None:
        print ("error: image not read from file \n\n")
        os.system("pause")
        return
    # end if
    out = np.zeros( noise_img.shape, np.uint8)
    imgGray = cv2.cvtColor( noise_img, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.medianBlur(imgGray, 5)
    cv2.GaussianBlur(imgGray, (5,5), 0)                    # blur
    gray = cv2.medianBlur(imgGray, 3)


    imgThresh = cv2.adaptiveThreshold(imgBlurred,
                                      255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV,
                                      11,
                                      2)



    imgThreshCopy = imgThresh.copy()

    npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,
                                                 cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_SIMPLE)

    for npaContour in npaContours:                             # for each contour
        contourWithData = ContourWithData()                                             # instantiate a contour with data object
        contourWithData.npaContour = npaContour                                         # assign contour to contour with data
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rect
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                    # get bounding rect info
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calculate the contour area
        allContoursWithData.append(contourWithData)                                     # add contour with data object to list of all contours with data
    # end for

    for contourWithData in allContoursWithData:
        if contourWithData.checkIfContourIsValid():
            validContoursWithData.append(contourWithData)
        # end if
    # end for

    validContoursWithData.sort(key = operator.attrgetter("intRectX"))         # sort contours from left to right

    strFinalString = ""

    for contourWithData in validContoursWithData:

        imgROI = imgThresh[contourWithData.intRectY: contourWithData.intRectY + contourWithData.intRectHeight,
                 # crop char out of threshold image
                 contourWithData.intRectX: contourWithData.intRectX + contourWithData.intRectWidth]

        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH,
                                            RESIZED_IMAGE_HEIGHT))  # resize image, this will be more consistent for recognition and storage

        npaROIResized = imgROIResized.reshape(
            (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # flatten image into 1d numpy array

        npaROIResized = np.float32(npaROIResized)  # convert from 1d numpy array of ints to 1d numpy array of floats
        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized,
                                                                         k=1)  # call KNN function find_nearest
        strCurrentChar = str(chr(int(npaResults[0][0])))  # get character from results
        if (strCurrentChar == 'o'):

            cv2.rectangle(noise_img,
                          (contourWithData.intRectX, contourWithData.intRectY),
                          (contourWithData.intRectX + contourWithData.intRectWidth,
                           contourWithData.intRectY + contourWithData.intRectHeight),
                          (0, 255, 0),
                          2)
            strFinalString = strFinalString + strCurrentChar
            cv2.putText(out, strCurrentChar,
                        (contourWithData.intRectX, contourWithData.intRectY + contourWithData.intRectHeight), 0, 1,
                        (0, 255, 0))

    # end for

    print ("\n" + strFinalString + "\n")

    cv2.imshow("imgTestingNumbers",  noise_img)
    cv2.imshow('out', out)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return


if __name__ == "__main__":
    main()



