import cv2
import numpy as np
import pytesseract
import imutils 
import csv
import pandas as pd



def contour(image):
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    contours,_=cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    return contours
def findCircle(cnts,img1):
    if len(cnts) == 0:
        print("No contour found!!")
    else:
        print(len(cnts))
        print("Starting circle search")
        img = img1.copy()
        circles = []
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle andSub1v4V
		# centroid
        for c in cnts:
            print("Inside Loop")
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            c_area = cv2.contourArea(c)
            c_area = c_area/((3.14)*radius*radius)
            if radius>10:
                print("radius > 3")
                if radius<50:
                    print("Radius < 50")
                    if c_area>0.65:
                        print("C_area > 0.65")
                        circles.append([int(x), int(y), radius, radius, "OMR", np.nan, 0])
                        cv2.circle(img, (int(x), int(y)), int(radius),(0,0,255), 1)
#                        cv2.circle(imgout,(int(x), int(y)), int(radius),(0,0,255), 2)
#                        cv2.circle(imgout, center, 5, (0, 0, 255), -1)
                        cv2.circle(img, center, 5, (0, 0, 255), -1)
#                        cv2.ellipse(img3,ellipse,(255,0,0),2)
                        
            else:
                print("Small Contours Found")
        cv2.imshow("contour", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("circle.png",img)
        print(len(circles))
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
"""



def linesp(img):
#    img1 = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 50,150, apertureSize = 3)
#    thres = 100
#    ret,gray = cv2.threshold(gray,thres,255,cv2.THRESH_BINARY)
    lines = cv2.HoughLinesP(canny, 1, np.pi/180, 80,None, 50, 1)
    return lines
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

    
for number in range(22,23):
    img = cv2.imread("test"+str(number)+".jpg")
    print("test"+str(number)+".png")
    width = 1000
    height = 50
    diff = 30
    img = imutils.resize(img, width=width)
    img3 = img.copy()
#    img = img[150:250,:]
    img1 = img.copy()
    img2 = img.copy()
    cv2.imshow("crop", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#"""Label Detection and Processing"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    gray = cv2.erode(threshold, None)
    gray = cv2.dilate(gray, None)
    
#    text = pytesseract.image_to_string(img)
#    print(text)
    text = pytesseract.image_to_boxes(gray)
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
        if w<height:
            if y_dist<height and x_dist<height:
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
    
    coordinates = []
    rectangle_image,rec_coordinate=detect_rectangles(img)
    for x,y in rec_coordinate:
        coordinates.append([x[0],x[1],y[0],y[1]])
    df_box = pd.DataFrame(coordinates, columns = ['X1','Y1','X2','Y2'])
    
    df_box = df_box.sort_values(by= ['Y1','X1']).reset_index(drop=True)
    print("DataFrame of Rectangle",df_box)
    x1,y1,x2,y2 = 0,0,0,0
    for row in df_box.itertuples():
        index_v = row[0]
        if row[1]<10 and row[2]<10:
            df_box = df_box.drop(row[0])        
        try:
            while(index_v in df_box.index and abs(y1-df_box.loc[index_v][1])<10):
                if index_v in df_box.index:
    #                    print(row[0]-1, y1,index_v, df.loc[index_v][1])
                    if abs(x1-df_box.loc[index_v][0])<diff and abs(y1-df_box.loc[index_v][1])<diff and abs(x2-df_box.loc[index_v][2])<diff and abs(y2-df_box.loc[index_v][3])<diff:
                        df_box = df_box.drop(index_v)
                    else:
                            pass
                    index_v+=1
        except Exception as e:
            print(e)
        try:
            index_curr = row[0]
            x1 = df_box.loc[row[0]][0]
            y1 = df_box.loc[row[0]][1]
            x2 = df_box.loc[row[0]][2]
            y2 = df_box.loc[row[0]][3]
        except:
            pass
    print("DataFrame of Rectangle",df_box)
    df_box = df_box.sort_values(by= ['Y1','X1']).reset_index(drop=True)
    
  
            
    
#"""Line detection and processing"""    
    line = linesp(img)
    horiz_lines = []
    if line is not None:
        for j in range(0, len(line)):
            l = line[j][0]
            if abs(l[1]-l[3])<diff:
                horiz_lines.append(line[j][0])

    df = pd.DataFrame(horiz_lines, columns=['X1','Y1','X2','Y2'])
    df = df.sort_values(by= ['Y1','X1']).reset_index(drop=True)
    filename = "Data"+str(number)+".csv"
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(df)
#    df.to_csv("Data"+str(number)+".csv")
#    print(df.to_csv(filename))
    x1,y1,x2,y2 = 0,0,0,0
    field_box = []
    index_curr = 0
    i=0
    for row in horiz_lines:
        cv2.line(img2,(row[0],row[1]),(row[2],row[3]),(0,0,255),2)
    cv2.imshow("lines", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("lines_detected.jpg", img2)
#    print(df)
    for row in df.itertuples():
        index_v = row[0]
        if row[0] == 35:
            print("Hello")
        try:
            while(index_v in df.index and abs(y1-df.loc[index_v][1])<10):
                if index_v in df.index:
#                    print(row[0]-1, y1,index_v, df.loc[index_v][1])
                    if abs(x1-df.loc[index_v][0])<diff and abs(y1-df.loc[index_v][1])<diff and abs(x2-df.loc[index_v][2])<diff and abs(y2-df.loc[index_v][3])<diff:
                        df = df.drop(index_v)
#                        print(df)
#                        print("one")
                    elif abs(y1-df.loc[index_v][1])<diff and (x1<=df.loc[index_v][0] and x2>=df.loc[index_v][2]):
                        df = df.drop(index_v)
#                        print("Two")
                    elif abs(y1-df.loc[index_v][1])<diff and (x1<=df.loc[index_v][0] and x2>=df.loc[index_v][0]):
                        field_box.pop()
                        if x2<df.iloc[index_v][2]:
                            field_box.append([x1, y1-height, df.loc[index_v][2], y1, "Field", np.nan, 0])
                            x2 = df.loc[index_v][2]
                            df.loc[index_curr] = [x1, y1, df.loc[index_v][2], y1]
                            df = df.drop(index_v)
                        else:
                            df = df.drop(index_v)
#                        print("Three")
                    elif abs(y1-df.loc[index_v][1])<diff and (x1>=df.loc[index_v][0] and x1<=df.loc[index_v][2]):
                        field_box.pop()
                        if x2>df.loc[index_v][2]:
                            field_box.append([df.loc[index_v][0], y1-height, x2, y1, "Field", np.nan, 0])
                            x1 = df.loc[index_v][0]
                            df.loc[index_curr] = [df.loc[index_v][0], y1, x2, y1]
                            df = df.drop(index_v)
                        else:
                            field_box.append([df.loc[index_v][0], y1-height, df.loc[index_v][2], y1, "Field", np.nan, 0])
                            df = df.drop(index_curr)
#                        print("Four")
                    elif abs(y1-df.loc[index_v][1])<diff and abs(x2-df.loc[index_v][0])<diff:
                        field_box.pop()
                        field_box.append([x1, y1-height, df.loc[index_v][2], y1, "Field", np.nan, 0])
                        x2 = df.loc[index_v][2]
                        df.loc[index_curr] = [x1, y1, df.loc[index_v][2], y1]
                        df = df.drop(index_v)
#                        print("Five")
                    elif abs(y1-df.loc[index_v][1])<diff and abs(x1-df.loc[index_v][2])<diff:
                        field_box.pop()
                        field_box.append([df.loc[index_v][0], y1-height, x2, y1, "Field", np.nan, 0])
                        df.loc[index_curr]= [df.loc[index_v][0], y1, x2, y1]
                        x1 = df.loc[index_v][0]
                        df = df.drop(index_v)
#                        print(df)
                    else:
                        pass
                index_v+=1
        except Exception as e:
            print(e)
        try:
            index_curr = row[0]
            x1 = df.loc[row[0]][0]
            y1 = df.loc[row[0]][1]
            x2 = df.loc[row[0]][2]
            y2 = df.loc[row[0]][3]
            field_box.append([x1,y1-height,x2,y2,"Field",np.nan,0])
        except:
            pass
            
        
    df = df.sort_values(by= ['Y1','X1']).reset_index(drop=True)
    print(df)
    print("Looop Restart")
#    for row in df.itertuples():
#        index_v = row[0]
#        if row[0] == 35:
#            print("Hello")
#        try:
#            while(index_v in df.index and abs(y1-df.loc[index_v][1])<10):
#                if index_v in df.index:
#                    print(row[0]-1, y1,index_v, df.loc[index_v][1])
#                    if abs(x1-df.loc[index_v][0])<diff and abs(y1-df.loc[index_v][1])<diff and abs(x2-df.loc[index_v][2])<diff and abs(y2-df.loc[index_v][3])<diff:
#                        df = df.drop(index_v)
##                        print(df)
#                        print("one")
#                    elif abs(y1-df.loc[index_v][1])<diff and (x1<=df.loc[index_v][0] and x2>=df.loc[index_v][2]):
#                        df = df.drop(index_v)
#                        print("Two")
#                    elif abs(y1-df.loc[index_v][1])<diff and (x1<=df.loc[index_v][0] and x2>=df.loc[index_v][0]):
#                        field_box.pop()
#                        if x2<df.iloc[index_v][2]:
#                            field_box.append([x1, y1-height, df.loc[index_v][2], y1, "Field", np.nan, 0])
#                            df.loc[index_curr] = [x1, y1, df.loc[index_v][2], y1]
#                            x2 = df.loc[index_v][2]
#                            df = df.drop(index_v)
#                        else:
#                            df = df.drop(index_v)
#                        print("Three")
#                    elif abs(y1-df.loc[index_v][1])<diff and (x1>=df.loc[index_v][0] and x1<=df.loc[index_v][2]):
#                        field_box.pop()
#                        if x2>df.loc[index_v][2]:
#                            field_box.append([df.loc[index_v][0], y1-height, x2, y1, "Field", np.nan, 0])
#                            df.loc[index_curr] = [df.loc[index_v][0], y1, x2, y1]
#                            x1 = df.loc[index_v][0]
#                            df = df.drop(index_v)
#                        else:
#                            field_box.append([df.loc[index_v][0], y1-height, df.loc[index_v][2], y1, "Field", np.nan, 0])
#                            df = df.drop(index_curr)
#                        print("Four")
#                    elif abs(y1-df.loc[index_v][1])<diff and abs(x2-df.loc[index_v][0])<diff:
#                        field_box.pop()
#                        field_box.append([x1, y1-height, df.loc[index_v][2], y1, "Field", np.nan, 0])
#                        df.loc[index_curr] = [x1, y1, df.loc[index_v][2], y1]
#                        x2 = df.loc[index_v][2]
#                        df = df.drop(index_v)
#                        print("Five")
#                    elif abs(y1-df.loc[index_v][1])<diff and abs(x1-df.loc[index_v][2])<diff:
#                        field_box.pop()
#                        field_box.append([df.loc[index_v][0], y1-height, x2, y1, "Field", np.nan, 0])
#                        df.loc[index_curr]= [df.loc[index_v][0], y1, x2, y1]
#                        x1 = df.loc[index_v][0]
#                        df = df.drop(index_v)
##                        print(df)
#                    else:
#                        pass
#                index_v+=1
#        except Exception as e:
#            print(e)
#        try:
#            index_curr = row[0]
#            x1 = df.loc[row[0]][0]
#            y1 = df.loc[row[0]][1]
#            x2 = df.loc[row[0]][2]
#            y2 = df.loc[row[0]][3]
#            field_box.append([x1,y1-height,x2,y2,"Field",np.nan,0])
#        except:
#            pass
    
    start = df.iloc[0]
    end = df.loc[df.last_valid_index()]
    count = 0
    for row in df_box.itertuples():
        for rows in df.itertuples():
            if row[0] in df_box.index and rows[0] in df.index:
                if abs(row[1]-rows[1])<diff and abs(row[2]-rows[2])<diff:
                    df = df.drop(rows[0])
                    count+=1
                elif abs(row[3]-rows[3])<diff and abs(row[4]-rows[4])<diff:
                    df = df.drop(rows[0])
                    count+=1
                else:
                    pass
    print(count)
    print(df)
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
#img = cv2.imread("Test1.jpg")
#img = imutils.resize(img, width = width)
#img1 = img.copy()

for row in df.itertuples():
    cv2.rectangle(img1, (row[1],row[2]-height),(row[3],row[4]),(0,0,255),1)
#    text = pytesseract.image_to_string(img[row[1]:row[3],row[0]:row[2]])
#    print(text)
cv2.imshow("Display", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("result.jpg", img1) 
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

