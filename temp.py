import cv2
import numpy as np
import pytesseract
from spellchecker import SpellChecker
import imutils 



def contour(img):
    #ratio = img.shape[0]/float(img.shape[1])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurr = cv2.GaussianBlur(gray, (5,5),0)
    thres = 170
    ret,gray = cv2.threshold(blurr,thres,255,cv2.THRESH_BINARY)
    #gray = cv2.Canny(blurr, 50,150, apertureSize = 3)

    print("Thresahold" + str(thres))
    cv2.imshow("Thres", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    contours = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    return contours
        
def boxDetect(c):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    if len(approx)==4:
        (X,Y,W,H) = cv2.boundingRect(approx)
        return X,Y,W,H
    else:
        return 0,0,0,0 
def lines(img):
    img1 = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 50,150, apertureSize = 3)
    cv2.imshow("canny", canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    lines = cv2.HoughLines(canny, 1, np.pi/180, 50)
    try:
        for r,theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = r*a
            y0 = r*b
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img1, (x1,y1), (x2,y2), (255,255,0),1)
        return img1
    except:
        return 1
def linesp(img):
    img1 = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 50,150, apertureSize = 3)
    cv2.imshow("canny", canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(img)[0]
    try:
        img1 = lsd.DrawSegments(img1,lines)
        return img1
    except:
        return 1
    
for i in range(2,15):
    img = cv2.imread("test"+str(i)+".png")
    print("test"+str(i)+".png")
    img = imutils.resize(img, width=700)
    spell = SpellChecker()
    contours = contour(img)
    try:
        """img1 = lines(img)
        cv2.imshow("img1", img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""
        print("ENtered")
        img1 = linesp(img)
        cv2.imshow("p", img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("Error Encountered")
    for c in contours:
        X,Y,W,H = boxDetect(c)
        if H !=0:
            cv2.rectangle(img, (X,Y),(X+W,Y+H), (0,255,0), 2)
            #cv2.drawContours(img,, -1, (0, 255, 0), 2)
        else:
            print("No Box Found")
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
#gray = cv2.GaussianBlur(gray, (3,3), cv2.BORDER_DEFAULT)
#gray = cv2.erode(gray, None, iterations=1)
    
#text = pytesseract.image_to_string(img)
#print(text)
#text = ' '.join(map(spell.correction, text.split()))

