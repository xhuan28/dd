import sys
import cv2
import numpy as np
import math
import string
import os
from PIL import Image

sizeh = 224 #the height of the input image to the net
sizew = 224 #the weight of the input image to the net
rotang = 5
filepath = './1train/'
numpath = './2num/'
croppath = './2crop/'
negpath = './3neg/'
pospath = './3pos/'
rotnegpath = './4negrot/'
rotpospath = './4posrot/'
gammapath = './5gammpos/'
g1 = 0.5
g2 = 2.2

def create_gammatable(g):
	gamma_table = [np.power(x / 255.0, g) * 255.0 for x in range(256)]
	gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
	return gamma_table

def gamma(img,gamma_table):
	img1 = np.zeros(img.shape, np.uint8)
	for i in range(sizeh):
		for j in range(sizew):
			(b,g,r) = img[i,j]
			b = gamma_table[b]
			g = gamma_table[g]
			r = gamma_table[r]
			img1[i,j] = (b,g,r)
	return img1

def gamma_image(file,gt1,gt2):
	img = cv2.imread(rotpospath+file)
	img1 = gamma(img,gt1)
	img2 = gamma(img,gt2)
	cv2.imwrite(gammapath+file[0:-4]+'_1.jpg',img)
	cv2.imwrite(gammapath+file[0:-4]+'_2.jpg',img1)
	cv2.imwrite(gammapath+file[0:-4]+'_3.jpg',img2)

#rotate the img for an angle,
def rotate(img,angle):
	img1 = img.convert('RGBA')
	rot = img1.rotate(angle,expand = 1)#keep the image in the graph
	rot = rot.resize((sizew,sizeh))
	bak = Image.new('RGBA',rot.size, (196,198,208,255)) #the background
	out = Image.composite(rot,bak,rot)
	return out.convert(img.mode)

#first rotate the image, then horizontally flip it
def rotate_flip(file,rotang):
	img = Image.open(pospath+file)
	count = 0
	for i in range(0,360,rotang):
		img1 = rotate(img,i)
		count += 1
		img1.save(rotpospath+file[0:-4]+'_'+str(count)+'.jpg')
		img2 = img1.transpose(Image.FLIP_TOP_BOTTOM)
		count += 1
		img2.save(rotpospath+file[0:-4]+'_'+str(count)+'.jpg')

#write the crop count on each pieces of the picture
def num_image(file):
	img = cv2.imread(filepath+file)
	img_w = img.shape[1]
	img_h = img.shape[0]
	num_w = img_w/sizew
	num_h = img_h/sizeh
	font = cv2.FONT_HERSHEY_SIMPLEX

	for i in range(1,num_w+1,1):
		cv2.line(img,(i*sizew,0),(i*sizew,img_h),(0,0,255),3)

	for i in range(1,num_h+1,1):
		cv2.line(img,(0,i*sizeh),(img_w,i*sizeh),(0,0,255),3)

	count = 0
	for i in range(num_w):
		for j in range(num_h):
			count = count+1
			print i,j,count
			cv2.putText(img, str(count),(i*sizew+sizew/2,j*sizeh+sizeh/2),font,2,(0,0,255),3)

	cv2.imwrite(numpath+file,img)

def crop_image(file):
	img = cv2.imread(filepath+file)
	img_w = img.shape[1]
	img_h = img.shape[0]
	num_w = img_w/sizew
	num_h = img_h/sizeh

	count = 0
	for i in range(num_w):
		for j in range(num_h):
			count = count+1
			#print i,j,count
			subimg = img[j*sizeh:(j+1)*sizeh,i*sizew:(i+1)*sizew]
			cv2.imwrite(croppath+file[0:-4]+'_'+str(count)+'.jpg',subimg)

	if img_w % sizew != 0:
		for j in range(num_h):
			count = count+1
			#print j,count
			subimg = img[j*sizeh:(j+1)*sizeh,num_w*sizew:img_w]
			cv2.imwrite(croppath+file[0:-4]+'_'+str(count)+'.jpg',subimg)

	if img_h % sizeh != 0:
		for i in range(num_w):
			count = count+1
			#print i, count
			subimg = img[num_h*sizeh:img_h,i*sizew:(i+1)*sizew]
			cv2.imwrite(croppath+file[0:-4]+'_'+str(count)+'.jpg',subimg)
		if img_w % sizew != 0:
			count = count+1
			#print count
			subimg = img[num_h*sizeh:img_h,num_w*sizew:img_w]
			cv2.imwrite(croppath+file[0:-4]+'_'+str(count)+'.jpg',subimg)
			

if __name__ == '__main__':
	#crop the image
	files = os.listdir(filepath)
	for file in files:
		num_image(file)
		crop_image(file)
	
	#rotate and flip
	files = os.listdir(pospath)
	for file in files:
		rotate_flip(file,rotang)
	
	
	#gamma transformation
	gamma_table1 = create_gammatable(g1)
	gamma_table2 = create_gammatable(g2)

	files = os.listdir(rotpospath)
	for file in files:
		gamma_image(file,gamma_table1,gamma_table2)
	
