# OpenCV Text Detection-Recognition-Translation

This project uses OpenCV to perform scene text detection, recognition, and translation on images by using the TextDetection and TextRecognition APIs from OpenCV.

Detection is completed using a pretrained [DBNet](https://github.com/MhLiao/DB) model found [here](https://drive.google.com/drive/folders/1T9n0HTP3X3Y_nJ0D1ekMhCQRHntORLJG). Recognition uses a pretrained model found [here](https://drive.google.com/drive/folders/1cTbQ3nuZG-EKWak6emD_s8_hHXWz7lAr?usp=sharing). Translation is relatively simple and uses the PyPi module [translate](https://pypi.org/project/translate/) and the MyMemoryAPI.

The script 'text_spotting.py' is capable of performing detection, recognition, and translation on images and video relatively quickly (on my laptop with no CUDA GPU, the pipeline can run at about 20 fps on my camera). There are a number of hardcoded detection/recognition model parameters (which can be changed) in the model loading functions contained in the utils file.