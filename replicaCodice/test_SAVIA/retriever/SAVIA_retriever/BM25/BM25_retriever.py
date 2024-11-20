from rank_bm25 import BM25Okapi
import numpy as np

class BM25Retriever():
    def __init__(self, corpus, top_n = 100, threshold = None):
#        print("init")
        self.corpus = corpus
        self.top_n = top_n
        self.threshold = threshold
        self.retriever = self.load_bm25_retriever(self.corpus)

    def load_bm25_retriever(self, corpus):
        tokenized_docs = [item.lower().split(" ") for item in corpus]
        bm25 = BM25Okapi(tokenized_docs)

        return bm25

    def retrieve(self, query):
        """
        returns top_n indices 
        and top_n similarities
        """

        query = query.lower()
#        texts = [item['titolo'] + " " + item['testo'] for item in corpus]

        tokenized_query = query.split(" ")
#        tokenized_query = [x.strip() for x in tokenized_query]
        sim = self.retriever.get_scores(tokenized_query)

        top_n_inds = np.argsort(sim)[::-1][0:self.top_n]
        top_n_sim = np.array([round(sim[top_n_ind], 5) for top_n_ind in top_n_inds])

        if self.threshold:
            top_n_inds = top_n_inds[top_n_sim > self.threshold]
            top_n_sim = top_n_sim[top_n_sim > self.threshold]

#            print(top_n_sim)
#            print(top_n_inds.shape)

        return top_n_inds, top_n_sim
    



