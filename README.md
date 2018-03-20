# Enron_sanwal
First machine learning project ever. 
This is my first attempt at making a functional machine learning model. I did this project as a final project for the Udacity Intro to Machine Learning class. This class was based around using Enron data to study concepts of machine learning. Enron was a gigantic conglomerate which went under, spectacularly as a result of systemic corporate corruption. This data was made public by FERC after their investigations.This data set has become a playground for novice machine learner like myself. 

For this final project, we had to engineer our features from the dataset and use those features to predict with certain accuracy, the people of interest(POI). We used the skills we gained throughout this course to do so.

The ENRON data and scripts associated with Udacity’s Intro to Machine Learning can all be found here.
https://github.com/udacity/ud120-projects

The features which I engineered were the ratio of email conversation any person has had with a POI with respect to all their email conversations. The intuition was that people who were suspected POIs would communicate with other POIs more often than not. 

I tried a few different algorithms to check the accuracy and precision of my predictions. The algorithms which i used are as follows:

Gaussian Naive-Bayes:
  The accuracy score of Gaussian NB is:  0.909090909091 and precision score is 0.826446280992

Decision Tree:
  The accuracy score for Decision Tree Classifer is 0.878787878788 and presicion score is  0.823863636364

Support Vector Machines: 
  The accuracy score for SVM is 0.909090909091 and presicion score is  0.826446280992

Random Forrest:
  The accuracy score for RF is 0.848484848485 and presicion score is  0.869122257053
  
  
You can check the code yourself and see how you like it! I am always up for criticism and comments! The script featureFormat contains a function that is called in test.py which contains the main code. 

I look forward to getting deeper into machine learning and enacting some of my own ideas, 




