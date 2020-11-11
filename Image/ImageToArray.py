from glob import glob
from os import listdir
import numpy as np
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
import shutil

def load_images(path, size=(256,256)):
    img_list = list()
    # enumerate filenames in directory, assume all are images
    for filename in listdir(path):
        if(".png" not in filename):
            continue
        # print(filename)
        # load and resize the image
        pixels = load_img(path + filename, target_size=size)
        # convert to numpy array
        pixels = img_to_array(pixels)
        img_list.append(pixels)
    return img_list

def makeTest(testPath,inputPath):
    files =listdir(inputPath)
    print(len(files))
    # enumerate filenames in directory, assume all are images
    for fileNumber in range(0,len(files),5):
        newPath = shutil.copy(inputPath + files[fileNumber],testPath + files[fileNumber])

def test():
    path = r'E:\EEE4022S\EEE4022S\Pipeline\Data\Output\\'
    path = path[:-1]
    testPath = path + 'Test' + '\\'
    inputPath = path + 'Input' + '\\'
    makeTest(testPath,inputPath)

def main():
    maxWidth = 200        
    path = r'E:\EEE4022S\EEE4022S\Pipeline\Data\Output\\'
    path = path[:-1]
    Input = []
    inputPath = path + 'Input' + '\\'
    Input = load_images(inputPath, size=(512, 512))
    Input_Array = asarray(Input)
    print(type(Input_Array))
    print(inputPath)
    iterCount = 25
    while(maxWidth>120):
        minWidth =maxWidth - 10
        outputSimple = []
        outputIPM = []
        testData  = []
        testPath = path + 'Test' + '\\'

        # makeTest(testPath,inputPath)



        # newPath = path + 'Simple' + '_'+str(maxWidth)+'_'+ str(minWidth)+'\\'
        # print(newPath)
        # outputSimple = load_images(newPath, size=(512,512))

        newPath = path + 'IPM' + '_'+str(maxWidth)+'_'+ str(minWidth)+'\\'
        print(newPath)
        outputIPM = load_images(newPath, size=(512,512))

        
        #   print(testPath)
        # testData = load_images(testPath, size=(512,512))

        # print('hi')
        # print(sign_list)
        # print(len(sign_list))
        p =r'E:\Documents\Drive\PixOutput\IPM\iter' + str(iterCount) +'\inputData'
        outputSimple_Array = asarray(outputSimple)
        print(type(outputSimple_Array))
        outputIPM_array = asarray(outputIPM)
        print(type(outputIPM_array))
        # load dataset
        print('Loaded: ', Input_Array.shape, outputSimple_Array.shape, outputIPM_array.shape)
        # save as compressed numpy array
        # filename1 = newPath + '\dataSimple_'+str(maxWidth)+'_'+ str(minWidth)+'.npz'
        # savez_compressed(filename1, Input_Array, outputSimple_Array)
        # print('Saved dataset: ', filename1)

        filename2 = p + '\dataIPM_'+str(maxWidth)+'_'+ str(minWidth)+'.npz'
        savez_compressed(filename2, Input_Array, outputIPM_array)
        print('Saved dataset: ', filename2)

        # filename3 = 'dataTest.npz'
        # savez_compressed(filename3, testData)
        # print('Saved dataset: ', filename3)
        maxWidth -= 10
        iterCount -=1


if __name__ == '__main__':
    # test()
    main()