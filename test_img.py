import shutil
import re
import os
import cv2
import sys
import numpy as np
import detect

sizeh = 224 #the height of the input image to the net
sizew = 224 #the weight of the input image to the net
fm_a = 1
weights = '/home/xiaojun/xinyao/objdet/caffe/models/objdet/objdet_iter_10000.caffemodel'
data_path = '/home/xiaojun/xinyao/objdet/dataset3/'
test_path = data_path + 'test_image/'
#label_file = None
label_file = data_path + 'label.txt'
#result_file = None
result_file = data_path + 'test_result.txt'

true_pos = 0.0
false_pos = 0.0
false_neg = 0.0
true_neg = 0.0
accuracy = 0.0
precision = 0.0
recall = 0.0
f_measure = 0.0

def crop_image(image,test_path):
	img = cv2.imread(test_path+image)
	subdir = test_path+image[0:-4]+'_subimg'
	img_h = img.shape[0]
	img_w = img.shape[1]
	num_h = img_h/sizeh
	num_w = img_w/sizew

	count = 0
	for i in range(num_w):
		for j in range(num_h):
			count = count+1
			#print i,j,count
			subimg = img[j*sizeh:(j+1)*sizeh,i*sizew:(i+1)*sizew]
			cv2.imwrite(subdir+'/'+image[0:-4]+'_'+str(count)+'.jpg',subimg)

	if img_w % sizew != 0:
		for j in range(num_h):
			count = count+1
			#print j,count
			subimg = img[j*sizeh:(j+1)*sizeh,num_w*sizew:img_w]
			cv2.imwrite(subdir+'/'+image[0:-4]+'_'+str(count)+'.jpg',subimg)

	if img_h % sizeh != 0:
		for i in range(num_w):
			count = count+1
			#print i, count
			subimg = img[num_h*sizeh:img_h,i*sizew:(i+1)*sizew]
			cv2.imwrite(subdir+'/'+image[0:-4]+'_'+str(count)+'.jpg',subimg)
		if img_w % sizew != 0:
			count = count+1
			#print count
			subimg = img[num_h*sizeh:img_h,num_w*sizew:img_w]
			cv2.imwrite(subdir+'/'+image[0:-4]+'_'+str(count)+'.jpg',subimg)

def test_image():
	files = os.listdir(test_path)
	for file in files:
		if re.search('subimg',file):
			shutil.rmtree(test_path+'/'+file)

	test_images = os.listdir(test_path)
	test_images.sort()
	dict = {}

	for image in test_images:
		dict[image] = 0
		subdir = test_path+image[0:-4]+'_subimg'
		os.mkdir(subdir)
		crop_image(image,test_path)
		os.chdir(subdir)
		sub_images = os.listdir(subdir)
		with open('test_image.txt','w') as f:
			for sub_image in sub_images:
				f.write(subdir+'/'+sub_image+' 0\n')
		os.chdir(data_path)
#		accuracy = 1
		_, accuracy = detect.eval_detect_net(weights,100,subdir+'/'+'test_image.txt')
		print image+' '+str(accuracy)
		if (accuracy != 1):
			dict[image] = 1
		#shutil.rmtree(subdir)	

	return dict

def eval(dict):
	file = open(label_file)
	
	global true_pos, false_pos, true_neg, false_neg
	global accuracy, precision, recall, f_measure
	count = 0.0
	for line in file:
		count += 1
#		print line, line[0:-4],line[-3:-2]
		test_val = dict[line[0:-3]]
		real_val = int(line[-2:-1])

		if real_val == 1:
			if test_val == 1:
				true_pos += 1
			else:
				false_neg += 1
		else:
			if test_val == 1:
				false_pos += 1
			else:
				true_neg += 1
	if count != 0:
		accuracy = (true_pos + true_neg) / count
		if true_pos != 0:
			precision = true_pos / (true_pos + false_pos)
			recall = true_pos / (true_pos + false_neg)
			f_measure = (fm_a^2 + 1)*precision*recall / ((fm_a^2)*(precision + recall))

def show_result(dict):
	if label_file is not None:
		file0 = open(label_file)
		file = []
		for line in file0:
			file.append(line)
		file.sort()
		print 'image test real'
		for line in file: 
			print line[0:-3]+'  '+str(dict[line[0:-3]])+'   '+line[-2:-1]
		print 'true_pos = '+str(true_pos)+', false_pos = '+str(false_pos)
		print 'false_neg = '+str(false_neg)+', true_neg = '+str(true_neg)
		print 'accuracy = '+str(accuracy)
		print 'precision = '+str(precision)
		print 'recall = '+str(recall)
		print 'f'+str(fm_a
)+'_measure = '+str(f_measure)
	else:
		print 'image test'
		for i in dict: 
			print i+' '+str(dict[i])

def save_result(dict):
	if label_file is not None:
		file0 = open(label_file)
		file = []
		for line in file0:
			file.append(line)
		file.sort()
		with open(result_file,'w') as f:
			f.write('image test real\n')
			for line in file: 
				f.write(line[0:-3]+'  '+str(dict[line[0:-3]])+'   '+line[-2:-1]+'\n')
			f.write('\n')
			f.write('true_pos = '+str(true_pos)+', false_pos = '+str(false_pos)+'\n')

			f.write('false_neg = '+str(false_neg)+', true_neg = '+str(true_neg)+'\n')
			f.write('accuracy = '+str(accuracy)+'\n')
			f.write('precision = '+str(precision)+'\n')
			f.write('recall = '+str(recall)+'\n')
			f.write('f'+str(fm_a)+'_measure = '+str(f_measure)+'\n')
	else:
		with open(result_file,'w') as f:
			f.write('image test\n')
			for i in dict: 
				f.write(i+' '+str(dict[i])+'\n')

if __name__ == '__main__':
	detect.caffe.set_mode_gpu()
	dict = test_image()
	if label_file is not None:
		eval(dict)
	if result_file is not None:
		save_result(dict)
	show_result(dict)


