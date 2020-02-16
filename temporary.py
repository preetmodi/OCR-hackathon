# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 16:00:55 2020

@author: Preet
"""
a = [1,2,3,4,5]
for i in range(len(a)):
    print(a[i])
    if len(a)>4:
        a.pop(1)
        