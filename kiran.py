import cv2
import numpy as np
import imutils
import pytesseract
def detect_rectangles(image):
    
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
    contours,_=cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    coordinate=[]
    co=0
    for cnt in contours:

        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.contourArea(cnt)>100:
            '''if 2000+co > cv2.contourArea(cnt) > co:         
                continue
                co=cv2.contourArea(cnt)'''# try changing the value in place of 2000 to get outer rectangles
            coordinate.append((approx[0][0],approx[2][0]))
            
    print(coordinate[0][0])# top-left coordinate of 1 rectangle
    print(coordinate[0][1])# bottom-right coordinate of 1 rectangle
    print(coordinate[1][0])# top-left coordinate of 2 rectangle
    print(coordinate[1][1])# bottom-right coordinate of 2 rectangle
    #and so on
    for i in range(len(coordinate)):
        cv2.circle(img,tuple(coordinate[i][0]),5,(0,255,0),5)
        cv2.circle(img,tuple(coordinate[i][1]),5,(0,255,0),5)
    return img,coordinate
def detect_circles(image):
    img=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1,img.shape[0]/64,
                               param1=100, param2=30,
                               minRadius=1, maxRadius=40)
    coordinate=[]
    for [x, y, r] in circles[0]:
        x=int(round(x))
        y=int(round(y))
        r=int(round(r))
        cv2.circle(image, (x, y),r, (0, 255, 0), 4)
        coordinate.append([[x-r,y-r],[x+r,y+r]])
    for i in range(len(coordinate)):
        cv2.rectangle(img,tuple(coordinate[i][0]),tuple(coordinate[i][1]),(0,255,0),2)
    return img,coordinate

image=cv2.imread("test20.jpg")
#image=imutils.resize(image, width = 1000)
rectangle_image,rec_coordinate=detect_rectangles(image)
#image=cv2.imread("Test1.jpg")  
#image=imutils.resize(image, width = 1000)
#for row in rec_coordinate:
#    text = pytesseract.image_to_string(image[row[0][1]:row[1][1],row[0][0]:row[1][0]])
#    print(text)
#    cv2.imshow("Boxes", image[row[0][1]:row[1][1],row[0][0]:row[1][0]])
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
cv2.imshow("rectangled image",rectangle_image)
cv2.imwrite("rec.png",rectangle_image)
circle_img,cir_coordinate=detect_circles(image)
cv2.imshow("rimage",circle_img)
cv2.imwrite("circle.png",circle_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("rec_coordinates: ",len(rec_coordinate))
print("circles_coordinates: ",len(cir_coordinate))
