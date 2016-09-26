import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import requests
from StringIO import StringIO
import urllib

import sys
import os

caffe_root = '/home/jonnatas/workspace_python/caffe/'
sys.path.insert(0, caffe_root + 'python')

print caffe_root

import caffe

if os.path.isfile(caffe_root + '/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print 'CaffeNet found.'

caffe.set_mode_cpu()

model_def = os.path.join(caffe_root, 'models', 'bvlc_reference_caffenet','deploy.prototxt')
model_weights = os.path.join(caffe_root, 'models','bvlc_reference_caffenet','bvlc_reference_caffenet.caffemodel')

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)

mu = np.load(os.path.join(caffe_root, 'python','caffe','imagenet','ilsvrc_2012_mean.npy'))
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

net.blobs['data'].reshape(50,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227

def options():
	print('1 - image')
	print('2 - url')
	print('3 - camera')
	print('4 - exit')

	return int(raw_input(''))

def detect(frame):
    height, width, depth = frame.shape

    # create grayscale version
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    # equalize histogram
    cv2.equalizeHist(grayscale, grayscale)

    # detect objects
    classifier = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml")

    # print classifier

    coords = []

    DOWNSCALE = 4
    minisize = (frame.shape[1]/DOWNSCALE,frame.shape[0]/DOWNSCALE)
    miniframe = cv2.resize(frame, minisize)
    faces = classifier.detectMultiScale(miniframe)
    if len(faces)>0:
        # print 'face detected!'
        for i in faces:
            x, y, w, h = [ v*DOWNSCALE for v in i ]

            coords.append((x,y,w,h))
            print x,y,w,h
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0))
    
    print(str(len(faces)) + ' faces detected.')
    print 'coords: '
    for coord in coords:
    	x,y,w,h = coord
    	print [(x,y), (x+w,y+h), (x+w, y), (x, y+h)]
    	print '\n'

    return frame

def get_image_url(url):
	cap = cv2.VideoCapture(url)
	ret,img = cap.read()

	return img

def get_image_camera():

	cap = cv2.VideoCapture(0)

	while(True):
	    # Capture frame-by-frame
	    ret, frame = cap.read()
	    img = frame.copy()
	    # Call the function
	    # frame = detect(frame)

	    # Display the resulting frame
	    cv2.imshow('frame',frame)
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	    	cap.release()#cv2.destroyAllWindows()
	        cv2.destroyAllWindows()
	        break

	cv2.imwrite("frame.jpg", frame)
	cv2.imwrite(os.path.join(caffe_root, 'examples', 'images','frame.jpg'), frame)

	return frame

def probability(image):
	transformed_image = transformer.preprocess('data', image)
	plt.imshow(image)
	plt.axis('off')

	# copy the image data into the memory allocated for the net
	net.blobs['data'].data[...] = transformed_image

	### perform classification
	output = net.forward()
	# print output

	output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

	# print 'predicted class is:', output_prob.argmax()

	labels_file = os.path.join(caffe_root, 'data','ilsvrc12','synset_words.txt')
	    
	labels = np.loadtxt(labels_file, str, delimiter='\t')

	dogs_file = os.path.join(caffe_root, 'data','ilsvrc12','dogs.txt')
	cats_file = os.path.join(caffe_root, 'data', 'ilsvrc12','cats.txt')

	labels_dogs = np.loadtxt(dogs_file, str, delimiter='\t')
	labels_cats = np.loadtxt(cats_file, str, delimiter='\t')

	top_inds = output_prob.argsort()[::-1][:]

	index = 0

	list_dogs = []
	probability_dogs = 0.0

	for i in labels:
		if i in labels_dogs:
			list_dogs.append((output_prob[index],labels[index]))
			probability_dogs += output_prob[index]
		index += 1

	index = 0

	list_cats = []
	probability_cats = 0.0

	for i in labels:
		if i in labels_cats:
			list_cats.append((output_prob[index],labels[index]))
			probability_cats += output_prob[index]
		index += 1

	list_dogs = sorted(list_dogs, reverse=True)
	list_cats = sorted(list_cats, reverse=True)

	for i in list_dogs:
		dprobability , dog_class = i
		print 'Dog synset ' + str(dog_class)
		print 'Probability ' + str(dprobability * 100.0) + "%"

	for i in list_cats:
		cprobability , cat_class = i
		print 'Cat synset ' + str(cat_class)
		print 'Probability ' + str(cprobability * 100.0) + "%"

	# clear_output()
	print '###############################################'
	print 'Feline probability: ' + str( (probability_cats) * 100.0 ) + "%"
	print 'Canine probability: ' + str( (probability_dogs) * 100.0 ) + "%"
	print 'Probable synset:', labels[output_prob.argmax()]
	print 'Probability: ' + str(output_prob[output_prob.argmax()] * 100.0) + "%"


if __name__ == '__main__':
	
	while True:
		option = options()


		if option == 1:
			print '1'

			image_name = raw_input('enter file name:\n')

			image = cv2.VideoCapture(image_name)
			ret, img = image.read()

			# detect(img)

			image_recognition = caffe.io.load_image(image_name)


			probability(image_recognition)
			image_detected = detect(img)

			plt.imshow(image_detected[:,:,::-1])
			plt.title('Detection result')
			plt.axis('off')
			plt.show()

		elif option == 2:
			
			image_url = raw_input('enter the image url:\n')
			image = get_image_url(image_url)

			urllib.urlretrieve(image_url, os.path.join(caffe_root, 'examples', 'images','url.jpg'))
			image_recognition = caffe.io.load_image(os.path.join(caffe_root, 'examples', 'images','url.jpg'))
			# print image

			probability(image_recognition)
			image_detected = detect(image)

			plt.imshow(image_detected[:,:,::-1])
			plt.title('Detection result')
			plt.axis('off')
			plt.show()


		elif option == 3:
			frame = get_image_camera()
			image_recognition = caffe.io.load_image(os.path.join(caffe_root, 'examples', 'images','frame.jpg'))
			probability(image_recognition)
			image_detected = detect(frame)

			plt.imshow(image_detected[:,:,::-1])
			plt.title('Detection result')
			plt.axis('off')
			plt.show()


		elif option == 4:
			break
