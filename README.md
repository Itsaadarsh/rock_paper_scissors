# Real time ROCK PAPER SCISSORS GAME
Hey guys this is my first project using OpenCV.It is basically a game of rock,paper,scissors against a computer.I first started with collecting images with rock,paper,scissors to train my model and luckly I found a dataset of 2500 images with separte training and testing images of rock,paper and scissors then I started cleaning my data and preprocessing all the images into numpy array of pixel values once that was done then I started with training the model : 
Model: "sequential_1"

=================================================================
conv2d_1 (Conv2D)            (None, 256, 256, 32)      896       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 128, 128, 32)      0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 128, 128, 32)      128       
_________________________________________________________________
dropout_1 (Dropout)          (None, 128, 128, 32)      0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 128, 128, 64)      18496     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 64, 64, 64)        0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 64, 64, 64)        256       
_________________________________________________________________
dropout_2 (Dropout)          (None, 64, 64, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 64, 64, 128)       73856     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 32, 32, 128)       0         
_________________________________________________________________
batch_normalization_3 (Batch (None, 32, 32, 128)       512       
_________________________________________________________________
dropout_3 (Dropout)          (None, 32, 32, 128)       0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 131072)            0         
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 393219    
=================================================================
Total params: 487,363
Trainable params: 486,915
Non-trainable params: 448
_________________________________________________________________
Once the model was trained it gave me around 97% accuracy which was really good.
The I got started with the game file in which I have to use OpenCV to read my webcam,draw bounding box to read my hand and basically put all my code together and start predicting the real time frames on my hand movement.
Soo ya this is basically it!
If you guys wanna try it just download the finalmodel.h5 file and run rps.py file on your system.
# THANKYOU
# SUPPORT AND FEEDBACKS
INSTAGRAM - @aadarshcodes
EMAIL - aadarsh.s2019@vitbhopal.ac.in / saadarsh362@gmail.com


