
from flask import Flask, render_template, url_for, request
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

filename = 'language_detection_model.pkl'
model = pickle.load(open(filename, 'rb'))
tfidf = pickle.load(open('transform.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
        return render_template('home.html')

@app.route('/predict',methods=['POST'])

def predict():
#        data = pd.read_csv('LanguageDetection.csv')

#       x = np.array(data['Text'])
#        y = np.array(data['Language'])

        ## Using TF-IDF Vectorizer + Logistic Regression
#       tfidf = TfidfVectorizer(
#            analyzer='char',
#            ngram_range=(2,4),
#            lowercase=True,
#            max_features=100000,
#            dtype=np.float32
#        )

#       X = tfidf.fit_transform(x)

#       pickle.dump(tfidf, open("transformf.pkl", "wb"))

 #       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#        model = LinearSVC( class_weight='balanced')

#       model.fit(X_train, y_train)

#       model.score(X_test,y_test)
#        user = input("Enter your text: ")
#        data = tfidf.transform([user]).toarray()
#        predicted = model.predict(data)
#        print(f"\nPredicted Language: {predicted}")

        if request.method == 'POST':
            message = request.form['message']
            data = tfidf.transform([message]).toarray()
            prediction = model.predict(data)
        return render_template('home.html',prediction=prediction, message = message) 

if __name__ == '__main__' :
        app.run(debug=True)


