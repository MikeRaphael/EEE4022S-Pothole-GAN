# import the necessary packages
import cv2  
import numpy as np
import math
from random import randrange
from astropy.modeling.models import Ellipse2D
from astropy.coordinates import Angle
from regions import PixCoord, EllipsePixelRegion
import matplotlib.pyplot as plt
import glob
import sys
from copy import deepcopy
import warnings

def importImage(path):   
    # Reading an image in default mode 
    return cv2.imread(path) 


def showImage(window_name,image):
    # Displaying the image  
    cv2.imshow(window_name, image)
    # allows us to see image
    # until closed forcefully
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getImageDimensions(image):
    # get dimensions of image
    dimensions = image.shape
    # height, width, number of channels in image
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2] 
    return height,width,channels  

def warpImage(img,width,height,pts1,pts2,inv):

    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    if(inv == 1):
        invmatrix =np.linalg.inv(matrix)
        output = cv2.warpPerspective(img,invmatrix,(width,height))
    else:
        output = cv2.warpPerspective(img,matrix,(width,height))
    return output

def fitEllipse(cc,axesLength,angle):
    x0, y0 = cc[0],cc[1]
    a, b = axesLength[0],axesLength[1]
    theta = Angle(angle, 'deg')
    e = Ellipse2D(amplitude=100., x_0=x0, y_0=y0, a=a, b=b, theta=theta.radian)
    y, x = np.mgrid[0:1000, 0:1000]
    center = PixCoord(x=x0, y=y0)
    reg = EllipsePixelRegion(center=center, width=2*a, height=2*b, angle=theta)
    patch = reg.as_artist(facecolor='none', edgecolor='red', lw=2)
    return reg

def makeEllipse(height,width,wSize,hSize,black,image,pts1,pts2,maxWidth,hSizemax,ccArr):
    angle = randrange(-15,15) 
    startAngle = 0
    endAngle = 360
    iterCount = hSize-1
    # print(pts1)
    # for i in pts1:
    #     if(i[0]>1392):
    #         i[0] = 1392
    #     elif(i[0]<0):
    #         i[0] = 0
    #     if(i[1]>512):
    #         i[1] = 512 
    #     elif(i[1]<0):
    #         i[1] = 0
    if(len(pts2) == 0):
        cc = getPosition(pts1)
    else:
        cc = getPosition(pts2)

    # print("cc",cc) 
    centre = True
    while(centre):
        ccCheck = deepcopy(cc)
        if(len(ccArr) == 0):
            break
        for i in ccArr:
            if((cc[1] + 2*hSizemax > i[1] and cc[1] - 2*hSizemax < i[1])):
                if((cc[0] + 2*maxWidth > i[0] and cc[0] - 2*maxWidth < i[0])):
                    if(len(pts2) == 0):
                        cc = getPosition(pts1)
                    else:
                        cc = getPosition(pts2)
        if(cc == ccCheck):
            centre = False
    # print("Worked", cc)
    ccArr.append(cc)
    cc = (555,389)
    roadB = 0
    roadG = 0
    roadR = 0
    count = 0
    # print(pts1[0][0])
    # for i in range(int(pts1[0][0]),int(pts1[3][0])):
    #     for j in range(int(pts1[0][1]),int(pts1[3][1])):
    #         # print(i,j)
    #         roadB += image[j,i][0] 
    #         roadG += image[j,i][1] 
    #         roadR += image[j,i][2] 
    #         count +=1
    # roadBGR = [int(roadB/count),int(roadG/count),int(roadR/count)]
    roadBGR = image[cc[1], cc[0]]
    centreBGR = [59,67,66]
    # centreBGR = [30,92,210] # Orange5
    # centreBGR = [145, 179, 210]
    # centreBGR = [61,95,117]
    color =(int(roadBGR[0]),int(roadBGR[1]),int(roadBGR[2]))
    axesLength = (wSize, hSize)
    reg = fitEllipse(cc,axesLength,angle)
    ellipse = cv2.ellipse(black, cc, axesLength, 
                angle, startAngle, endAngle, color, -1)  
    for i in range(iterCount):
        axesLength = (wSize-i, hSize-i)  
        B = int(roadBGR[0] - i*(roadBGR[0] - centreBGR[0])/(iterCount-1))
        G = int(roadBGR[1] - i*(roadBGR[1] - centreBGR[1])/(iterCount-1))
        R = int(roadBGR[2] - i*(roadBGR[2] - centreBGR[2])/(iterCount-1))
        color = (B,G,R) 
        thickness = cv2.FILLED
        ellipse = cv2.ellipse(ellipse, cc, axesLength, 
                angle, startAngle, endAngle, color, thickness) 
    return ellipse


def addImage(ellipse,image):
    kernel  = np.ones((3,3), np.float32)/9
    smoothed = cv2.filter2D(ellipse,-1,kernel)
    smoothed =ellipse
    # Creating kernel 
    # kernel = np.ones((5, 5), np.uint8) 
  
    # Using cv2.erode() method  
    # kernel  = np.ones((3,3), np.float32)/9
    # smoothed = cv2.filter2D(ellipse,-1,kernel)
    # kernel = np.ones((5, 5), np.uint8) 
    # smoothed = cv2.erode(smoothed, kernel)  
    # ksize = (30,30)
    # cv2.blur(smoothed, ksize, cv2.BORDER_DEFAULT) 
    # smoothed = ellipse
    # I want to put logo on top-left corner, So I create a ROI
    rows,cols,channels = ellipse.shape
    roi = image[0:rows, 0:cols ]
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(smoothed,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 60, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(smoothed,smoothed,mask = mask)
    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)
    image[0:rows, 0:cols ] = dst
    # showImage("bg",img1_bg)
    # showImage("fg",img2_fg)
    # showImage("dst",dst)
    return dst

def addClearImage(image2,image):
    # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    # image = cv2.filter2D(image, -1, kernel)
    rows,cols,channels = image2.shape
    roi = image[0:rows, 0:cols ]
    img2gray = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 30, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    img2_fg = cv2.bitwise_and(image2,image2,mask = mask)
    # img2_fg = cv2.GaussianBlur(img2_fg, (3, 3), 0)
    dst = cv2.add(img1_bg,img2_fg)
    # alpha = 0.9
    # beta = 1-alpha
    # dst = cv2.addWeighted(img1_bg, alpha, img2_fg, beta, 0.0)
    image[0:rows, 0:cols ] = dst
    # showImage("bg",img1_bg)
    # showImage("fg",img2_fg)
    # showImage("dst",dst)
    return dst


def showImage(window_name,image):
    # # Displaying the image  
    # final = cv2.add(image,smoothed)
    cv2.imshow(window_name, image)
    # allows us to see image
    # until closed forcefully
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def getPosition(pts1):
    # w = randrange(width//5,2*width//3)
    # h = randrange(2*height//3, height)
    
    if(pts1[0][0]> pts1[1][0]):
        w1 = pts1[1][0]
        w2 = pts1[0][0]
    else:
        w1 = pts1[0][0]
        w2 = pts1[1][0]
    if(pts1[0][1] > pts1[2][1]):
        h1 = pts1[2][1]
        h2 = pts1[0][1]
    else:
        h1 = pts1[0][1]
        h2 = pts1[2][1]
    try:
        w = randrange(w1, w2)
        h = randrange(h1,h2)
        # print(w1,w2,w)
    except:
        # print("getPosition")
        TopLeft = (600,300)
        TopRight = (780, 200)
        BottomLeft = (300, 512)
        BottomRight = (875, 512)
        w = randrange(math.ceil((TopLeft[0] + BottomLeft[0])/2),math.ceil((TopRight[0] + BottomRight[0])/2))
        h = randrange(math.ceil((BottomRight[1] + BottomLeft[1])/2),math.ceil((TopRight[1] + TopLeft[1])/2))
    center_coordinates = (w, h) 
    return center_coordinates


def canny_edge_detector(image): 
      
    # Convert the image color to grayscale 
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  
    ret, mask = cv2.threshold(gray_image, 175, 255, cv2.THRESH_BINARY)
    # plt.subplot(1,2,1)
    # plt.imshow(mask,cmap = 'gray')
    # plt.subplot(1,2,2)
    # plt.imshow(gray_image,cmap = 'gray')
    # plt.show()
    # Reduce noise from the image 
    blur = cv2.GaussianBlur(mask, (5, 5), 0)  
    canny = cv2.Canny(blur, 50, 150) 
    return canny 


def region_of_interest(image): 
    height = image.shape[0] 
    width  = image.shape[1]
    # [(Bottom Left), (Bottom Right), (Top Right), (Top Left)] 

    # 2011_09_26_drive_0002_extract
    #  polygons = np.array([ 
    #     [(100, height), (1200, height), (600, 200), (300,350)] 
    #     ]) 

    TopLeft = (600,300)
    TopRight = (780, 200)
    BottomLeft = (300, height)
    BottomRight = (875, height)
    polygons = np.array([ 
        [BottomLeft, BottomRight, TopRight, TopLeft] 
        ]) 
    mask = np.zeros_like(image) 
      
    # Fill poly-function deals with multiple polygon 
    cv2.fillPoly(mask, polygons, 255)  
      
    # Bitwise operation between canny image and mask image 
    masked_image = cv2.bitwise_and(image, mask)  
    return masked_image 


def create_coordinates(image, line_parameters): 
    slope, intercept = line_parameters 
    y1 = image.shape[0] 
    y2 = int(y1 * (3 / 5)) 
    x1 = int((y1 - intercept) / slope) 
    x2 = int((y2 - intercept) / slope) 
    return np.array([x1, y1, x2, y2]) 


def average_slope_intercept(image, lines): 
    left_fit = [] 
    right_fit = [] 
    for line in lines: 
        x1, y1, x2, y2 = line.reshape(4) 
        parameters = np.polyfit((x1, x2), (y1, y2), 1)  
        slope = parameters[0] 
        intercept = parameters[1] 
        if slope < 0: 
            left_fit.append((slope, intercept)) 
        else: 
            right_fit.append((slope, intercept)) 
              
    left_fit_average = np.average(left_fit, axis = 0) 
    right_fit_average = np.average(right_fit, axis = 0) 
    left_line = create_coordinates(image, left_fit_average) 
    right_line = create_coordinates(image, right_fit_average) 
    return np.array([left_line, right_line]) 


def display_lines(image, lines): 
    line_image = np.zeros_like(image) 
    if lines is not None: 
        for x1, y1, x2, y2 in lines: 
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10) 
    return line_image 

def getCorners(lines):
    corners = []
    if lines is not None: 
        for x1, y1, x2, y2 in lines: 
            c1 = [x2, y2]
            c2 = [x1, y1]
            corners.append(c1)
            corners.append(c2)
    return corners

def laneDetect(frame,imageName,imgArr):
    canny_image = canny_edge_detector(frame) 
    plt.imshow(canny_image, cmap = "gray")
    plt.show()
    cropped_image = region_of_interest(canny_image) 
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100,  
                            np.array([]), minLineLength =  20,  
                            maxLineGap = 10) 
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            averaged_lines = average_slope_intercept(frame, lines) 
        except np.RankWarning:
            imgArr.append(imageName)
    line_image = display_lines(frame, averaged_lines) 
    pts1 = getCorners(averaged_lines)
    pts1 = np.float32([pts1[0], pts1[2],pts1[1],pts1[3]])
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)  
    # plt.subplot(1,2,1)
    # plt.imshow(cropped_image)
    # plt.subplot(1,2,2)
    # plt.imshow(combo_image)
    # plt.show()
    pathWrite = r"C:\Users\MikeR\OneDrive\Documents\EEE4022S\EEE4022S\Pipeline\LaneDetect"
    out = pathWrite + "\\" + imageName
    cv2.imwrite(out, combo_image) 
    return pts1

def getImages(path):
    images = [[file,cv2.imread(file)] for file in glob.glob(path)]
    return images

def getData():
    dataList = []
    ##########
    DataSet 1
    dataName = "2011_09_26_drive_0001_extract"
    TL = [529,276]		
    TR = [710, 276]
    BL = [0, 512]		
    BR = [862, 512]
    dataList.append([dataName,TL,TR,BL,BR])
    #DataSet 2
    dataName = "2011_09_26_drive_0009_extract"
    TL = [600,282]		
    TR = [727, 282]
    BL = [600, 512]		
    BR = [958, 512]
    dataList.append([dataName,TL,TR,BL,BR])


    #DataSet 3
    dataName = "2011_09_26_drive_0011_extract"
    TL = [600,288]		
    TR = [777, 288]
    BL = [342, 495]		
    BR = [928, 512]
    dataList.append([dataName,TL,TR,BL,BR])
    #DataSet 11
    dataName = "2011_10_03_drive_0042_extract"
    TL = [582, 315]		
    TR = [750, 315]
    BL = [284, 512]		
    BR = [806, 512]
    dataList.append([dataName,TL,TR,BL,BR])
    #DataSet 12
    dataName = "2011_09_26_drive_0051_extract"
    TL = [576,281]		
    TR = [700, 281]
    BL = [300, 512]		
    BR = [945, 512]
    dataList.append([dataName,TL,TR,BL,BR])
    ##############
    #DataSet 4
    dataName = "2011_09_26_drive_0002_extract"
    TL = [534,256]		
    TR = [612, 257]
    BL = [2, 495]		
    BR = [890, 512]
    dataList.append([dataName,TL,TR,BL,BR])
    #DataSet 5
    dataName = "2011_09_26_drive_0015_extract"
    TL = [527, 293]		
    TR = [715, 293]
    BL = [168, 512]		
    BR = [893, 512]
    dataList.append([dataName,TL,TR,BL,BR])
    #DataSet 6
    dataName = "2011_09_26_drive_0028_extract"
    TL = [623, 298]		
    TR = [710, 298]
    BL = [350, 512]		
    BR = [770, 512]
    dataList.append([dataName,TL,TR,BL,BR])
    #DataSet 7
    dataName = "2011_09_26_drive_0029_extract"
    TL = [575, 293]		
    TR = [715, 293]
    BL = [245, 512]		
    BR = [844, 512]
    dataList.append([dataName,TL,TR,BL,BR])
    #DataSet 8
    dataName = "2011_09_26_drive_0032_extract"
    TL = [580, 293]		
    TR = [715, 293]
    BL = [344, 512]		
    BR = [860, 512]
    dataList.append([dataName,TL,TR,BL,BR])
    #DataSet 9
    dataName = "2011_09_26_drive_0070_extract"
    TL = [600, 293]		
    TR = [715, 293]
    BL = [284, 512]		
    BR = [830, 512]
    dataList.append([dataName,TL,TR,BL,BR])
    #DataSet 10
    dataName = "2011_09_26_drive_0084_extract"
    TL = [586, 309]		
    TR = [715, 293]
    BL = [391, 512]		
    BR = [830, 512]
    dataList.append([dataName,TL,TR,BL,BR])
    

    return dataList

def main():
    if len(sys.argv) >1:	
        for arg in sys.argv:   
            if (arg=='1'):
                option = "Simple"
            else:
                option = "IPM"
    else:
        option = "IPM"
    
    dataList = getData()
    maxWidth = 60
    # print(dataList)
    while(maxWidth>50):
        count =0
        for data in dataList:
            
            path = r'E:\EEE4022S\EEE4022S\Pipeline\Data\\'[:-1] + data[0] + r'\*.png'
            pathWrite = r'E:\EEE4022S\EEE4022S\Pipeline\Data\\'[:-1] + data[0] +r'\\'[:-1]
            print(pathWrite)
            TL = data[1]
            TR = data[2]
            BL = data[3]
            BR = data[4]

            
            pathWrite = pathWrite + 'Output\\' + option
            f= open(pathWrite + "\\" +  "ErrorImages.txt","w+")
            images = getImages(path)
            # i = images[170]
            imgArr = []

            for i in images:
                if(count%9==0):
                    # numPotholes = randrange(1,4)
                    numPotholes = 1
                    # print(numPotholes)
                    image = i[1]
                    imageName = i[0][-14:]
                    img = image.copy()
                    frame = image.copy() 
                    try:
                        height,width,channels = getImageDimensions(img)
                        try:
                            pts1 = laneDetect(frame,imageName,imgArr)
                            # pts1 = np.float32([TL,TR,BL,BR])
                        except Exception as e: 
                            print(e)
                            print(imageName)
                            # continue  
                            # TL TR BL BR
                        pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
                        if (option=='Simple'):
                            output = image.copy()
                        elif (option=='IPM'):
                            output = warpImage(img,width,height,pts1,pts2,0)
                            # showImage("IPM", output) 
                        # for x in range(0,4):
                        #     cv2.circle(img, (pts1[x][0],pts1[x][1]),5,(0,0,255),cv2.FILLED)
                        # showImage("CircleTest",img)
                        h,w,channels = getImageDimensions(output)
                        ellipseArr =[]
                        ccArr = []
                        for i in range(numPotholes):
                            if (option=='Simple'):
                                black = np.zeros((h,w,3), np.uint8)
                                minWidth = maxWidth-10
                                hSizemax = math.ceil(40*maxWidth/100)
                                wSize = randrange(minWidth,maxWidth)
                                scalar = randrange(40,45)
                                hSize = math.ceil(scalar*wSize/100)
                                ellipse = makeEllipse(h,w,wSize,hSize,black,output,pts1,[],maxWidth,hSizemax,ccArr)
                                ellipseArr.append(ellipse)
                            elif (option=='IPM'):
                                black = np.zeros((h,w,3), np.uint8)                
                                # maxWidth = 100
                                # minWidth = 65
                                minWidth = maxWidth-105
                                hSizemax = math.ceil(55*maxWidth/100)
                                wSize = randrange(minWidth,maxWidth)
                                scalar = randrange(40,55)
                                hSize = math.ceil(scalar*wSize/100)
                                ellipse = makeEllipse(h,w,wSize,hSize,black,output,pts1,pts2,maxWidth,hSizemax,ccArr)
                                ellipseArr.append(ellipse)
                        dst = addImage(ellipseArr[0],output)
                        for i in range(1,len(ellipseArr)):
                            dst = addImage(ellipseArr[i],dst)
                        if (option=='Simple'):
                            pass
                        elif (option=='IPM'):
                            # showImage("Image with " + str(numPotholes) + " Potholes",output)
                            output = warpImage(dst,width,height,pts1,pts2,1)
                            dst = addClearImage(output,img)
                        # showImage("Image with " + str(numPotholes) + " Potholes",dst)
                        # if(imageName not in imgArr):
                        imagePath = pathWrite + "\\" + imageName
                        # print(imagePath)
                        
                        cv2.imwrite(imagePath, dst) 

                        imageName = "0000000" + str(count) + ".png" # ???
                        outWrite = r'E:\EEE4022S\EEE4022S\Pipeline\Data\\'[:-1] +r'\\'[:-1]+ 'Output\\' + option+"_"+str(maxWidth)+"_"+str(minWidth)
                        inWrite = r'E:\EEE4022S\EEE4022S\Pipeline\Data\\'[:-1] +r'\\'[:-1]+ 'Output\\' + "Input"
                        duplicateImage  = inWrite + "\\" + imageName
                        imagePath = outWrite + "\\" + imageName
                        cv2.imwrite(imagePath, dst) 
                        cv2.imwrite(duplicateImage, image) 

                        
                    except Exception as e: 
                        print(e)
                        f.write(imageName + "\n")
                count +=1
            f.close() 
        print(str(maxWidth)+"_"+str(minWidth) + "Complete")
        maxWidth -=10
        

    

if __name__ == '__main__':
    main()