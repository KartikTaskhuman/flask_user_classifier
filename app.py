from flask import Flask, jsonify, request
import numpy as np
from sklearn.externals import joblib
import s3fs

fs = s3fs.S3FileSystem(anon=False, key='AKIAILLLLVE5PJOLO2JA', secret='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')

count_vect=joblib.load(fs.open("taskhuman-models/v1/count_vect.pkl", 'rb'))
tf_transformer=joblib.load(fs.open("taskhuman-models/v1/tf_transformer.pkl", 'rb'))
clf=joblib.load(fs.open("taskhuman-models/v1/clf.pkl", 'rb'))

app = Flask(__name__)

def print_model_results(query):
	querylist=[query]
	X_new_counts = count_vect.transform(querylist)
	X_new_tfidf = tf_transformer.transform(X_new_counts)
	probs=clf.predict_proba(X_new_tfidf)
	n=1
	best_n = np.argsort(probs)[:,:-n-1:-1]
	return clf.classes_[best_n][0]

@app.route('/predict', methods=['GET', 'POST'])
def index():
    query_data = request.get_json()
    query= query_data['query']
    get_top_recs=print_model_results(query)
    return jsonify({"category":get_top_recs[0]})

@app.route('/categories', methods=['GET'])
def get_stores():
	return jsonify({'classes': clf.classes_.tolist()})


app.run(debug=True)
