import re
import string
import spacy
import string
import re
#from stop_words import get_stop_words


nlp = spacy.load("it_core_news_sm")
#stopwords = get_stop_words('italian')

def remove_punctuation(text):
    out_text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    
    return out_text

def lemmatization(text):

    doc = nlp(text)

    # Extract lemmas from the analysed words
    lemmi = [token.lemma_ for token in doc]

    out_text = " ".join(lemmi)

    return out_text

"""
def remove_stopwords(text):

#    exclusions = '|'.join(stopwords)
#    out_text = re.sub(exclusions, '', text)
#    out_text = ""

#    for text in corpus:
    tmp = text.split(' ')
    tmp = [x.strip() for x in tmp]

    for stopword in stopwords:
        if stopword in tmp:
            tmp.remove(stopword)

    out_text = " ".join(tmp)

    return out_text
"""