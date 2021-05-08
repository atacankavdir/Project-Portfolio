from flask import Flask, request, render_template, url_for
from sklearn.feature_extraction.text import TfidfVectorizer 
import pickle 
import re 
from string import punctuation, digits 


app = Flask(__name__)

@app.route("/sentiment",methods=['POST']) 
def home():
	vec = open("models/model_sentiment.pkl",'rb')
	loaded_model = pickle.load(vec)
	vcb = open("models/vocab_sentiment.pkl", 'rb')
	loaded_vocab = pickle.load(vcb) 

	txt= request.args['text']
	example = txt

		#lower string 
	example = example.lower() 

	#remove numbers
	example =example.replace('\n', ' ') 

	#remove email adress
	example =re.sub('[a-zA-z0-9_.]+@[a-zA-Z0-9-_.]+', ' ', example) 

	#removeIP address 
	example =re.sub('((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', ' ', example) 

	#remove punctaitions and special chracters 
	example =re.sub('[^\W\S]', ' ', example) 

	#remove numbers
	example =re.sub('\d', ' ', example) 

	examples = [example]

	count_vect = TfidfVectorizer(
	    analyzer = 'word', 
	    ngram_range = (1,2),
	    max_features = 50000,
	    max_df= 0.6,
	    use_idf = True,
	    norm = 'l2', 
	    vocabulary = loaded_vocab
	)

	x_count = count_vect.fit_transform(examples)

	predicted = loaded_model.predict(x_count)

	if predicted == 1:
	    return 'Positive'
	elif predicted == 0:
		return 'Negative'
