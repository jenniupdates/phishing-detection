# Phishing Detection Model
Experimenting with hybrid models to detect phishing emails on online platforms.

Done for SMU mod.

## Project's Inspiration and Description
Phishing scams exist even today, with losses amounting to over $660 million in just 2022. 
As scammers are constantly evolving their phishing tactics coupled with the increase in online activities by the general public, we wanted to invent an improved model with better detection and classification of potential phishing scams, so as to protect society and our loved ones.

Generally, phishing emails detection can be divided into two types: the blacklist method and the machine-learning method. The blacklist checks the email address/url against a blacklist of known phish.
Although blacklist is simple and efficient, a paper which analysed the effectiveness of phishing blacklists concluded that it was ineffective against fresh feed. Which is why we will be turning to the machine learning method.

What we did different from most studies out there who applied machine learning models is that:
1. trained on a larger dataset where 11000 sample data will be used
2. tested different models and compared their results
3. factored in sentiment analysis in the prediction
4. combined different models together to predict phishing emails

## Project Methodoloy
First, we break down the email into 3 components: Text Message, Sentiment of the Email, and URLs of the email.
Then we will conduct analysis on each of them, train them on different models and identify the best models for each component.
After that, we will combine the models together to form an Ensemble Model. The Ensemble Model with be used to test on a foreign dataset. 
Finally, we will get the performance of the Ensemble Model and compare the results with that of other models respectively.

## Some Results
![email text analysis results](/images/email_text_results.png?raw=true "email text analysis results")
![email text sentiment subjectivity score results](/images/email_text_sentiment_results.png?raw=true "email text sentiment subjectivity score analysis results")
![url results](/images/url_results.png?raw=true "url analysis results")