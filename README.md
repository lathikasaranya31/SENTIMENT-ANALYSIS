# SENTIMENT-ANALYSIS

*COMPANY* : CODTECH IT SOLUTIONS

*NAME* : LATHIKA SARANYA

*INTERN ID* : CT06DF926

*DOMAIN* : DATA ANALYTICS

*DURATION* : 6 WEEKS

*MENTOR* : NEELA SANTHOSH

# DESCRIPTION OF THE TASK DONE

# The tool used
# Python – core language for programming and data analysis

# Jupyter Notebook – for writing and running the code interactively

# NLTK (Natural Language Toolkit) – for text preprocessing (stop words, lemmatization)

# scikit-learn (sklearn) – for TF-IDF vectorization, model training, and evaluation

# pandas – for data handling and manipulation

# matplotlib & seaborn – for visualizing the confusion matrix

# The editing platform is fully implemented using Jupyter Notebook, a popular browser-based interface for Python programming. It allows users to combine code, output, markdown documentation, and visualizations in a single interactive environment.
# This Sentimental Analysis can be widely used in real world industries like E-commerce, Social Media Monitoring, Political Analysis, Customer Support and Brand Management.
# The objective of this task is to perform Sentiment Analysis on textual data using Natural Language Processing (NLP) techniques. For this project, we used a dataset of tweets titled task4_full_twitter_sentiment_dataset.csv. This dataset contained over 1,000 real-world tweets, each labeled as either Positive, Negative, or Neutral.
# We began by loading the dataset and performing text preprocessing to clean the data. This involved Lowercasing all text to ensure uniformity,Removing URLs, special characters, numbers, and emojis and Tokenizing the text (breaking sentences into individual words)
# Removing stop words like “the”, “is”, “and” that do not carry sentiment ,Lemmatizing words to convert them into their base form. These cleaned tweets were stored in a new column called Cleaned_Tweet.
# Next, we transformed the text into numerical format using TF-IDF Vectorization (Term Frequency-Inverse Document Frequency). This step converts each tweet into a vector of numerical features based on the importance of each word relative to the rest of the corpus.
# We then split the dataset into training and testing sets (80% training, 20% testing) to build and validate the sentiment classification model. For the model, we used Logistic Regression, a widely-used algorithm for classification problems. It is simple, interpretable, and performs well on text classification tasks.
# After training the model on the training data, we made predictions on the test data and evaluated the results using Classification Report and Confusion Matrix by using heatmaps.

# The preprocessing phase had a major impact on model accuracy. Properly cleaning and lemmatizing the text helped reduce noise and allowed the model to focus on meaningful patterns. TF-IDF proved to be an effective vectorization method for this task, balancing common and rare words.

# The Logistic Regression model was chosen for its simplicity and speed. However, in real-world applications, more advanced models like Naive Bayes, SVM, or even deep learning (LSTM, BERT) can further improve performance.
# This task hepls in focusing on importance of data cleaning in NLP, Feature extraction techniques for text and Model training, testing, and evaluation pipelines.

# This task was a practical and insightful exercise in the application of NLP and machine learning for real-world sentiment analysis. It highlighted the power of combining text preprocessing, feature engineering, and classification to derive actionable insights from unstructured data. This project mirrors the kind of work done by data analysts and ML engineers in industry today, especially in marketing, customer service, and social media analytics.

# FINAL OUTPUT OF THE TASK 
![Image](https://github.com/user-attachments/assets/1fe0b367-acab-4ba1-b950-8aa710e6c336)
