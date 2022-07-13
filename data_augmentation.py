import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

source = 'Data/Train'

destination = 'Data/AugmentedTrain'

output_name = lambda class_name, img_name, id : '{destination}/{class_name}/{id}_{img_name}'.format(destination=destination,
                                                                            class_name=d,
                                                                            img_name=f, id=indx)

def rotate(img, theta):
    theta = theta / 180 * np.pi

    T = np.array([[1, 0, img.shape[1]/2],[0, 1, img.shape[0]/2], [0,0,1]])

    H = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    H = np.matmul(T,np.matmul(H, np.linalg.pinv(T)))
    return cv.warpPerspective(img, H, dsize=(img.shape[1], img.shape[0]))


angles = [0]
try:
    os.mkdir(destination)
except:
    pass
for d in os.listdir(source):
    print('class ' + d)
    if d[0] != '.':
        os.mkdir(destination + '/' + d)
        for f in os.listdir(source +'/'+ d):
            if f[0] != '.':
                original = cv.imread(source + '/' + d + '/' + f)
                indx = 0
                cropped = []
                h, w = original.shape[:2]
                h1 = h // 10
                w1 = w // 10
                cropped.append(original)
                # cropped.append(original[h1:,w1:,:])
                # cropped.append(original[h1:,:-w1,:])
                cropped.append(original[:-h1,w1:,:])
                cropped.append(original[:-h1,:-w1,:])

                for cropped_image in cropped:
                    for theta in angles:
                        img = rotate(cropped_image, theta)
                        if theta != 0:
                            img = img[25:-25,25:-25,:]
                    
                        cv.imwrite(output_name(class_name=d, img_name=f, id=indx), img)
                                                                
                        # img = img[:,::-1,:]
                        # cv.imwrite(output_name(class_name=d,img_name=f, id=indx+1), img)

                        indx += 2
