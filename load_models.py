import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from models import clean_df

string = """Hi Yin Shan,

Thank you so much for your confirmation!

Good luck with your interviews :)

Thank you!

Regards
Grace"""

test_df = {'Text':[string], 'Class': [0]}
test_df = pd.DataFrame(data=test_df)
test_df = clean_df(test_df) # process include download nltk stopwords, v long --> how to predownload?

xtest = test_df["Text_Filtered_String"]
ytest = test_df["Class"]

with open('./models/text_nb_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)
xtestidf = vectorizer.transform(xtest)

with open('./models/text_nb.pkl', 'rb') as file:
    nb = pickle.load(file)
predictions = nb.predict(xtestidf)

print('accuracy: %s%%' % round(accuracy_score(ytest,predictions)*100,0))