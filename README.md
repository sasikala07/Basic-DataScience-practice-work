# DATASCIENCE_PORTFOLIO

Repository containing portfolio of data science project completed by me for self learning and training purposes.Presented in the form of iPython Notebooks.

### End-End ML Project

PART - 1 : Model Building and hosting local API

1. Data Preparation 
2. Machine Learning Modelling 
3. Model Evaluation
4. Export Trained Model
5. LOCAL REST API with Flask web-server
6. Create a website for predicing marriage age calling REST API

PART - 2 : Deploying Public API to AWS EC2 server and launch website service

1. Spin up an EC2 server
2. Configure EC2 with security group and private key
3. Install libraries and dependencies on the EC2 server
4. Move trained model and app.py flask files to EC2 (winscp)
5. Configure flaskapp.wsgi file and Apache vhost file
6. Restart apache webserver and Check API status
6. Launch a website with domain name and host webpage.

- [Marriage_Prediction](https://github.com/sasikala07/DataScience_portfolio/tree/master/ML_Project) :
Model build with **RandomForest Regresion algorithm** and predicted the year of marriage.Deployed API to **AWS EC2 server** and launch website service. [Website Link](http://3.140.249.198/).Given below is the image that shows the website.

![Screenshot from 2021-12-07 11-13-17](https://user-images.githubusercontent.com/72785420/145705801-57c20395-8c27-4dd1-bd2b-ce7a1349995e.png)


### ML_Micro_Projects
- [HealthCare_Prediction](https://github.com/sasikala07/DataScience_portfolio/blob/master/ml_micro_proj/healthcare_stroke_detection.ipynb) :
A model to predict whether the pateint likely to get stroke or not.Identified the best **F1 Score** model from different agorithms to get accurate result in prediction.perform data visualization using **matplotlib and seaborn**.

- [Recommendation_System](https://github.com/sasikala07/DataScience_portfolio/blob/master/ml_micro_proj/Imdb_movies_recommendation_collaborative_and%20content_based_filtering.ipynb):
A model to recommend similar movies using both **Collaborative Filtering and Content Based Filtering** recommendation system.Done NLP **TFIDF Vectorize**.
