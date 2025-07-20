# Spam Detector using Machine Learning

This project is a text classification system that identifies whether a given message (SMS or email) is spam or not spam (ham). It uses several machine learning algorithms and natural language processing techniques to preprocess, vectorize, and classify the input text efficiently.

This repository is based on the original project by CampusX (linked below) but has been significantly modified to include enhanced preprocessing steps, feature engineering, better modularity, and comparative model evaluation. The aim was to build a more robust, scalable version of a spam classifier for real-world usage and experimentation.

### Yes the deployment is still left ik 

## Features

- **Text Preprocessing**  
  Cleans the messages by:
  - Converting to lowercase
  - Removing punctuation and special characters
  - Removing stopwords
  - Applying stemming using NLTK

- **Feature Engineering**  
  Additional features are created to improve model understanding:
  - `num_characters`: total characters in a message
  - `num_words`: total words
  - `num_sentences`: total sentences (using NLTK's sentence tokenizer)

- **Vectorization with TF-IDF**  
  Uses `TfidfVectorizer` from scikit-learn to convert cleaned text into numerical vectors, improving upon simpler Bag-of-Words methods by accounting for the importance of terms across the dataset.

- **Multiple Train-Test Splits**  
  Tests models with different `random_state` values to analyze how splits impact performance and ensure robust generalization.

- **Model Training & Evaluation**  
  Multiple classifiers are trained and evaluated:
  - Gaussian Naive Bayes
  - Support Vector Machine (SVC)
  - Random Forest
  - Logistic Regression

  Evaluation metrics include:
  - Accuracy
  - Precision
  - Confusion Matrix

- **Model Comparison**  
  Results from all classifiers are stored in DataFrames and sorted by metrics (like precision) to identify top performers. Separate evaluations are done after feature scaling and after limiting the max features for TF-IDF.

- **Scalability and Modularity**  
  The notebook is organized so new models, transformations, or feature engineering techniques can be easily integrated.

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/sn1kyy/spam-detector.git
cd spam-detector
```
2. Install the required packages:

```bash
pip install -r requirements.txt
```
3. Run the notebook:

```bash
jupyter notebook spam-detector.ipynb
```


## Credits

This project is a modified and improved version of the [CampusX SMS Spam Classifier](https://github.com/campusx-official/sms-spam-classifier/blob/main/sms-spam-detection.ipynb), originally created by CampusX on [YouTube](https://www.youtube.com/watch?v=YncZ0WwxyzU&t=4269s&ab_channel=CampusX).

### Key Improvements over the Original:
- Added engineered features: `num_characters`, `num_words`, `num_sentences`
- Switched from CountVectorizer to TF-IDF for better feature representation
- Used multiple classifiers with automated performance comparison
- Modularized model evaluation and comparison pipeline
- Enhanced visualizations and sorted model evaluation metrics



 

