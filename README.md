# PaDiM-Anomaly-Detection
A simple script of padim provides an easy way to achieve anomaly detection 

## Installation:  

●python=3.6  
●torch=1.4.0  
●opencv-python=4.2.0.34  

## Introduction of the mfcc_analysis.py:

1. In main, the function "train_by_extract_feat" is to compute both covariance and mean matrices for training data "J2901_3_UniformLight-OK226.jpg"  
2. Take efficientnet-B0 for instance, the result will be shown as below:  
![image](https://github.com/ChengWeiGu/PaDiM-Anomaly-Detection/blob/main/feat_maps.jpg)    
3. The function "test_by_anomaly_detection" in main is to test the input image "J2901_3_UniformLight-NG971.jpg"   
4. To consider different threshold, the result will be like this:  
![image](https://github.com/ChengWeiGu/PaDiM-Anomaly-Detection/blob/main/result.jpg)  
