# PaDiM-Anomaly-Detection
A simple script of padim provides an easy way to achieve anomaly detection 

## Installation:  

●python=3.6  
●torch=1.4.0  
●opencv-python=4.2.0.34  

## Introduction of the mfcc_analysis.py:

1. The function "train_by_extract_feat" in main can help extract the feature maps from a given backbone. Also, both covariance and mean matrices will be calculated and saved  
2. Take efficientnet-B0 for instance, the result will be shown as below:  
![image](https://github.com/ChengWeiGu/mfcc_analysis/blob/main/demo.jpg)    
3. The function "test_by_anomaly_detection" in main is to test the input image     
4. To consider different threshold, the result will be like this:  
![image](https://github.com/ChengWeiGu/mfcc_analysis/blob/main/demo2.jpg) 
