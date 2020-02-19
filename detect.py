import numpy as np
import cv2
import pytesseract
import csv
import pandas as pd

with open('data20.csv','r') as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)
pd.read_csv('data20.csv')
#print(df)