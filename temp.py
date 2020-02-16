import cv2
import numpy as np
import pytesseract
import imutils 
import csv
import pandas as pd



def contour(img):
    #ratio = img.shape[0]/float(img.shape[1])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurr = cv2.GaussianBlur(gray, (5,5),0)
    thres = 100
    ret,gray = cv2.threshold(blurr,thres,255,cv2.THRESH_BINARY)
    #gray = cv2.Canny(blurr, 50,150, apertureSize = 3)

    print("Thresahold" + str(thres))
    cv2.imshow("Thres", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    contours = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    return contours
def findCircle(cnts,img1):
    if len(cnts) == 0:
        print("No contour found!!")
    else:
        print("Starting circle search")
        img = img1.copy()
        circles = []
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle andSub1v4V
		# centroid
        for i in range(len(cnts)):
            print("Inside Loop")
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            c_area = cv2.contourArea(c)
            c_area = c_area/((3.14)*radius*radius)
            if radius>3:
                if radius<25:
                    if c_area>0.65:
                        circles.append([int(x), int(y), radius, radius, "OMR", np.nan, 0])
                        cv2.circle(img, (int(x), int(y)), int(radius),(0,0,255), 1)
#                        cv2.circle(imgout,(int(x), int(y)), int(radius),(0,0,255), 2)
#                        cv2.circle(imgout, center, 5, (0, 0, 255), -1)
                        cv2.circle(img, center, 5, (0, 0, 255), -1)
#                        cv2.ellipse(img3,ellipse,(255,0,0),2)
                        cv2.imshow("contour", img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
            else:
                print("Small Contours Found")
            cnts.remove(c)
        return circles


"""
            
        cv2.drawContours(mask, [c], -1, 0, -1)
        img1 = cv2.bitwise_and(img1, img1, mask=mask)
        
        
		#Calculate ratio of area to evaluate confidence
        e_area = 0
		#Ellipse cannot be fitted always so try for that or else we go with circle
        try:
            ellipse = cv2.fitEllipse(c)
            (a,b),(ma,mb), angle = ellipse
            e_area = cv2.contourArea(c)
            e_area = 4*e_area/(3.14*ma*mb)
            if radius>3 and radius<25:
                if c_area >0.6 and e_area > 0.81:
                    img = img1.copy()
                    cv2.circle(img, (int(x), int(y)), int(radius),255, 2)
                    cv2.circle(img1,(int(x), int(y)), int(radius),(0,0,255), 2)
            else:
                print("Small Contours Found")
                break
        except:
			#A = []
            print("Small ellipse found")
                    
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
    lines = cv2.HoughLinesP(canny, 1, np.pi/180, 500)
    try:
        for i in range(0,len(lines)):
            print(lines[i][0])
            r,theta = lines[i][0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = r*a
            y0 = r*b
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img1, (x1,y1), (x2,y2), (0,0,255),1)
        return img1
    except:
        print("No line found")
        return 1"""

def linesp(img):
#    img1 = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 50,150, apertureSize = 3)
#    thres = 100
#    ret,gray = cv2.threshold(gray,thres,255,cv2.THRESH_BINARY)
    lines = cv2.HoughLinesP(canny, 1, np.pi/180, 80,None, 50, 1)
    return lines
    
for number in range(16,21):
    img = cv2.imread("test"+str(number)+".png")
    print("test"+str(number)+".png")
    img = imutils.resize(img, width=700)
#    img = img[150:250,:]
    img1 = img.copy()
    cv2.imshow("crop", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#"""Label Detection and Processing"""
    
    text = pytesseract.image_to_string(img)
#    print(text)
    text = pytesseract.image_to_boxes(img)
    data = text.split('\n')
    text = []
    label_box = []
    value = ""
#    print(data)

    for j in range(len(data)):
        text.append(data[j].split(" "))
        
    data = ""
    X1,Y1 = int(text[0][1]),int(text[0][2])
    X2,Y2 = int(text[0][3]),int(text[0][4])
#    print(text)
    for i in range(len(text)):
#        print(abs(Y2-int(text[i][4])), abs(X2-int(text[0][1])))
#        print(text[i][0])
        
        data+=text[i][0]
        h = int(text[i][4])-int(text[i][2])
        w = int(text[i][3])-int(text[i][1])
        y_dist = abs(Y2-int(text[i][4]))
        x_dist = abs(X2-int(text[i][1]))
#        print(text[i][0],w,x_dist,y_dist)
        if w<30:
            if y_dist<20 and x_dist<20:
    #            X1,Y1 = int(text[i][1]),int(text[i][2])
                X2,Y2 = int(text[i][3]),int(text[i][4])
    #            print(text[i][0])
                value+= text[i][0] 
            else:
                if len(value)>1:
                    label_box.append([X1,Y1,X2,Y2,"Label",value,"0"])
                X1,Y1 = int(text[i][1]),int(text[i][2])
                X2,Y2 = int(text[i][3]),int(text[i][4])
    #            print(text[i][0],"First")
                value = ""
                value+=text[i][0]
    label_box.append([X1,Y1,X2,Y2,"Label",value,"0"])
#    print(data)
#    print(label_box)


#"""Line detection and processing"""    
    line = linesp(img)
    horiz_lines = []
    height = 30
    if line is not None:
        for j in range(0, len(line)):
            l = line[j][0]
            
            if abs(l[1]-l[3])<5 or abs(l[0]-l[2]<5):
                horiz_lines.append(line[j][0])

    df = pd.DataFrame(horiz_lines, columns=['X1','Y1','X2','Y2'])
    df = df.sort_values(by= ['Y1','X1']).reset_index(drop=True)
    x1,y1,x2,y2 = 0,0,0,0
    field_box = []
    i=0
    print(df)
    for row in df.itertuples():
        if row[0] in df.index:
            print(row[0])
            if abs(x1-row[1])<5 and abs(y1-row[2])<5:
                df = df.drop(row[0])
                print("Row Deleted",row[0])
            elif abs(y1-row[2])<5 and (x1<=row[1] and x2>=row[3]):
                df = df.drop(row[0])
                print("Row Deleted",row[0])
            elif abs(y1-row[0])<5 and (x1<=row[1] and x2>=row[1]):
                field_box.pop()
                field_box.append([x1, y1-20, row[3], y1, "Field", np.nan, 0])
                df = df.drop(row[0])
                print("Row Deleted",row[0])
            elif abs(y1-row[0])<5 and (x1>=row[1] and x1<=row[3]):
                field_box.pop()
                if x2>row[3]:
                    field_box.append([row[1], y1-20, x2, y1, "Field", np.nan, 0])
                else:
                    field_box.append([row[1], y1-20, row[3], y1, "Field", np.nan, 0])
                df.drop(row[0])
                print("Row Deleted",row[0])
            elif abs(y1-row[2])<5 and abs(x2-row[3])<5:
                field_box.pop()
                field_box.append([x1, y1-20, row[3], y1, "Field", np.nan, 0])
                df = df.drop(row[0])
                print("Row Deleted",row[0])
            elif abs(y1-row[2])<5 and abs(x1-row[3])<5:
                field_box.pop()
                field_box.append([row[1], y1-20, x2, y1, "Field", np.nan, 0])
                df = df.drop(row[0])
                print("Row Deleted",row[0])
            elif abs(row[2]-row[4])>10:
                df.drop(row[0])
            else:
                index_v = row[0]
                try:
    #                print(abs(y1-df.iloc[index_v][1]), df.iloc[index_v][1])
                    while(index_v < df.last_valid_index() and abs(y1-df.loc[index_v][1])<10):
                        if index_v in df.index:
                            print(row[0]-1, y1,index_v, df.loc[index_v][1])
                            if abs(x1-df.loc[index_v][0])<5 and abs(y1-df.loc[index_v][1])<5:
                                df = df.drop(index_v)
                                print(df)
                                print("one")
                            elif abs(y1-df.loc[index_v][1])<5 and (x1<=df.loc[index_v][0] and x2>=df.loc[index_v][2]):
                                df = df.drop(index_v)
                                print("Two")
                            elif abs(y1-df.loc[index_v][1])<5 and (x1<=df.loc[index_v][0] and x2>=df.loc[index_v][0]):
                                field_box.pop()
                                field_box.append([x1, y1-20, df.loc[index_v][2], y1, "Field", np.nan, 0])
                                df = df.drop(index_v)
                                print("Three")
                            elif abs(y1-df.loc[index_v][1])<5 and (x1>=df.loc[index_v][0] and x1<=df.loc[index_v][2]):
                                field_box.pop()
                                if x2>df.loc[index_v][2]:
                                    field_box.append([df.loc[index_v][0], y1-20, x2, y1, "Field", np.nan, 0])
                                else:
                                    field_box.append([df.loc[index_v][0], y1-20, df.loc[index_v][2], y1, "Field", np.nan, 0])
                                df = df.drop(index_v)
                                print("Four")
                            elif abs(y1-df.loc[index_v][1])<5 and abs(x2-df.loc[index_v][0])<5:
                                field_box.pop()
                                field_box.append([x1, y1-20, df.loc[index_v][2], y1, "Field", np.nan, 0])
                                df = df.drop(index_v)
                                print("Five")
                            elif abs(y1-df.loc[index_v][1])<5 and abs(x1-df.loc[index_v][2])<5:
                                field_box.pop()
                                field_box.append([df.loc[index_v][0], y1-20, x2, y1, "Field", np.nan, 0])
                                df = df.drop(index_v)
                                print("six")
                            else:
                                pass
                        index_v+=1
                except:
                    print("Index Out of Bounds")
                try:
                    x1 = df.loc[row[0]][0]
                    y1 = df.loc[row[0]][1]
                    x2 = df.loc[row[0]][2]
                    y2 = df.loc[row[0]][3]
                    field_box.append([x1,y1-20,x2,y2,"Field",np.nan,0])
                except:
                    pass
            
    print(df)
    for row in field_box:
        cv2.line(img1, (row[0],row[1]),(row[2],row[3]),(0,255,0),2)
    cv2.imwrite("img"+str(number)+".png",img1)
    with open('data'+str(number) + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["X1", "Y1", "X2", "Y2", "Type", "Value", "Group"])
        writer.writerows(label_box)
        writer.writerows(field_box)
        
    cnt = contour(img1)
    circles = findCircle(cnt, img1)

    cv2.imshow("p", img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#    for c in contours:
#        X,Y,W,H = boxDetect(c)
#        if H !=0:
#            cv2.rectangle(img, (X,Y),(X+W,Y+H), (0,255,0), 2)
#            cv2.drawContours(img,, -1, (0, 255, 0), 2)
#        else:
#            print("No Box Found")
#    cv2.imshow("img", img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

    
#gray = cv2.GaussianBlur(gray, (3,3), cv2.BORDER_DEFAULT)
#gray = cv2.erode(gray, None, iterations=1)
    
#text = pytesseract.image_to_string(img)
#print(text)
#text = ' '.join(map(spell.correction, text.split()))

