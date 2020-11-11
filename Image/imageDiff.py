import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir


def diff_images(path1, path2):
    img_list = list()
    # enumerate filenames in directory, assume all are images
    imageCount = 0
    totalCount = 0
    print(path1)
    for filename in listdir(path1):
        if(".png" not in filename):
            continue
        totalCount +=1
        img1Loc = path1 +"\\"+ filename
        img2Loc =path2 +"\\"+ filename
        
        # load and resize the image
        img1 = cv2.imread(img1Loc)
        img2 = cv2.imread(img2Loc)
        # convert to numpy array
        diff = cv2.absdiff(img1, img2)
        # mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        # th = 1
        # imask =  mask>th
        # canvas = np.zeros_like(img2, np.uiimg1Locnt8)
        # canvas[imask] = img2[imask]
        # # print(diff)
        count =0
        countMax = 100
        for x in diff:
            if(count>countMax):
                break
            for i in x:
                if(count>countMax):
                    break
                if(sum(i)>35):
                    # print(i)
                    if(all(z >= 23 for z in i)):
                        count +=1 
                # if(sum(i)>35):
                #     if(all(z >= 23 for z in i)):
                #         count +=1 
                # if(count>countMax):
                #     break
                    
        print(filename,img1Loc,count)
        # plt.title(str(count)+ "      "+ filename)
        # plt.imshow(diff)
        # plt.show()   
        if(count>countMax):
            print(count)
            
            # plt.subplot(1,2,1)
            # plt.title( "Original")
            # plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
            # plt.axis('off')
            # plt.subplot(1,2,2)
            # plt.title( "Generated")
            # plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
            # plt.axis('off')
            # plt.show()
            # plt.title( "Difference in " + filename +" files")
            # plt.imshow(diff)
            # plt.axis('off')
            # plt.show()
            imageCount +=1
            img_list.append(filename + '\n')
            if(imageCount%5==0):
                print(filename, imageCount)
    percent = 100*imageCount/totalCount
    print(imageCount, percent, "%")
    print(img_list)
    print()
    return imageCount, percent, img_list


iterOption ='IPM'
start = 17
end = 17
outputList = []
for i in range(start, end+1):
    iterCount = "iter"+ str(i)
    path1 = r'E:\Documents\Drive\PixOutput\\'[:-1] + iterOption + '\\' + iterCount + '\\test\original\\'[:-1]
    path2 = r'E:\Documents\Drive\PixOutput\\'[:-1] + iterOption + '\\' + iterCount + '\\test\\'[:-1]
    imageCount, success, imageList = diff_images(path1, path2)
    outputList.append([iterCount, success])
#[:-1]
    # Writing to file 
    with open(path2 +"\\"+ "results.txt", "w") as file1: 
        # Writing data to a file 
        print(path2 +"\\"+ "results.txt")
        file1.write(str(imageCount)+"\n") 
        file1.write(str(success) + " %"+ "\n") 
        file1.writelines(imageList) 
print(outputList)
print("COMPLETE!!")
# 19 -71-32