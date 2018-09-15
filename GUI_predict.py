from __future__ import division, print_function, absolute_import
#imports
import os
import tkinter
import tkinter.filedialog
from PIL import Image
#imports for core
import os
import tflearn
import tensorflow as tf
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import h5py
from tflearn.data_utils import build_hdf5_image_dataset

###preprocess
# Building 'AlexNet'
network = input_data(shape=[None, 227, 227, 3])
network = conv_2d(network, 48, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 128, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 192, 3, activation='relu')
network = conv_2d(network, 192, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 2048, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 2048, activation='relu')##
network = dropout(network, 0.5)
network = fully_connected(network, 5, activation='softmax')

network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)
model = tflearn.DNN(network)
model.load("cacoon.tfl.ckpt-8016")
NAMES = ['合格','瘪','黄斑','畸','双宫']

##tools
#labels&buttons
def build_label(text):
	name = tkinter.Label(root, text=text)
	name.pack()
	
def build_button(text, command=None):
	button=tkinter.Button(root, text=text, command=command)
	button.pack()
	return button
#openbutton behavior
def open():
	notification = tkinter.Label(root,text = '')
	filename = tkinter.filedialog.askopenfilename()
	if filename != '':
		notification.config(text="已选择文件："+filename)
	else:
		notification.config(text='未选择任何文件')
	notification.pack()

	image = Image.open(filename).resize((227, 227))
	img = np.asarray(image)
	img = img[np.newaxis, :]
	X = img
	saying=str(filename+'为:'+NAMES[model.predict(X).argmax()]+'茧')
	build_label(saying)

	nw=tkinter.Toplevel()
	new_filename=os.path.splitext(filename)[0]+'.gif'
	nw.title(new_filename)
	Image.open(filename).save(new_filename)
	img1_gif = tkinter.PhotoImage(file=new_filename)
	label_img1 = tkinter.Label(nw, image=img1_gif)
	label_img1.pack()
	nw.mainloop()
	
#building root-window
root = tkinter.Tk()
root.title('Cacoon-Classification')
root.resizable(True, True)
windowWidth = 960
windowHeight=640
screenWidth, screenHeight = root.maxsize()
geometryParam='%dx%d+%d+%d'%(windowWidth, windowHeight, (screenWidth-windowWidth)/2, (screenHeight-windowHeight)/2)
root.geometry(geometryParam)

#building utilities
button_select = tkinter.Button(root, text='选择',command=open)
button_select.pack()


#start looping
root.mainloop()
