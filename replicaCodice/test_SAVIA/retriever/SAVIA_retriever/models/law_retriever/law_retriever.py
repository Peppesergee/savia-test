#import faiss
#import os
from pymongo import MongoClient
#from langchain_text_splitters import TokenTextSplitter
#import numpy as np
#import sys
#sys.path.append("./models")
#from BM25.BM25_retriever import BM25Retriever
import time
#from vector_store_nb.helper_functions import generate_law_wording
from SAVIA_retriever.models.helper_functions import find_best_similarity, load_index
from SAVIA_retriever.models.law_retriever.helper_functions_law_retriever import load_BM25_retriever, get_law_documents, add_repealing_laws, load_reranker
#from models.law_retriever.reranker import Reranker

class LawRetriever():
    def __init__(self, configs, client = "mongodb://mongo_db:27017", db_SAVIA = "SAVIA", db_SAVIA_NLU = "SAVIA_NLU", 
                 vector_store_folder = "../../../../../SAVIA_vector_stores"):

            self.configs = configs
            if self.configs['verbose']:
                print("loading law retriever")

            self.client = MongoClient(client)
            self.db_SAVIA = self.client[db_SAVIA]
            self.db_SAVIA_NLU = self.client[db_SAVIA_NLU]
            self.coll_leggi_regionali = self.db_SAVIA["leggiRegionali"]
            self.BM25_retriever, self.list_all_laws = load_BM25_retriever(self.coll_leggi_regionali, 
                                                                        top_n = self.configs['retrievers']['law_retriever']['reranker']['top_n_BM25'])
            
            if self.configs['retrievers']['law_retriever']['reranker']['use_reranking']:
                print("loading law re-ranker")
#                self.reranker = Reranker(self.configs)
                self.reranker = load_reranker(self.configs)

            embedding_model_name = self.configs['models']['embedding_model']
            self.coll_leggi_regionali_wording, self.leggi_regionali_wording_index = load_index("leggiRegionaliWording", self.db_SAVIA_NLU, 
                                                                                               vector_store_folder, embedding_model_name)

            self.coll_leggi_regionali_summary, self.leggi_regionali_summary_index = load_index("leggiRegionaliSummary", self.db_SAVIA_NLU, 
                                                                                               vector_store_folder, embedding_model_name)

            self.coll_leggi_regionali_HQ_on_law, self.leggi_regionali_HQ_on_law_index = load_index("leggiRegionaliHQonLaw", self.db_SAVIA_NLU, 
                                                                                               vector_store_folder, embedding_model_name)

            self.coll_leggi_regionali_articles, self.leggi_regionali_articles_index = load_index("leggiRegionaliArticles", self.db_SAVIA_NLU, 
                                                                                               vector_store_folder, embedding_model_name)

            self.coll_leggi_regionali_articles_summaries, self.leggi_regionali_articles_summaries_index = load_index("leggiRegionaliArticlesSummaries", 
                                                                                                        self.db_SAVIA_NLU, 
                                                                                                        vector_store_folder, embedding_model_name)
            
            self.coll_leggi_regionali_HQ_on_articles, self.leggi_regionali_HQ_on_articles_index = load_index("leggiRegionaliHQonArticles", 
                                                                                                        self.db_SAVIA_NLU, 
                                                                                                        vector_store_folder, embedding_model_name)




    def get_laws(self, question, question_embedding, question_chunks_embeddings, entities_anno):
        """
        Funzione principale per la ricerca delle leggi.
        Effettua la ricerca delle leggi in tre fasi: diciture, BM25 + re-ranking, similarità semantica. 
        Procede allo stadio successivo solo se non ha trovato nulla.
        """

        law_list = []

        #retrive by matching law wordings
        law_list += self.find_laws_by_wording(question_chunks_embeddings)

        #retrive by re-ranking if active
        use_reranking = self.configs['retrievers']['law_retriever']['reranker']['use_reranking']

        if len(law_list) == 0 and use_reranking:
#        if True:
            law_list += self.find_laws_by_reranking(question)

        #retrive by semantic similarity
        if len(law_list) == 0:
            law_list += self.find_laws_by_semantic_similarity(question_embedding)

#        print(law_list)

        #rimuovi dalle entities l'anno della legge
        anno_leggi = [x['anno'] for x in law_list]

        for anno_legge in anno_leggi:
            if anno_legge in entities_anno:
                entities_anno.remove(anno_legge)

        entities_anno = [int(x) for x in entities_anno if x.isdigit()]

#        print("entities_anno", entities_anno)

        #aggiunge le legge abroganti di ogni legge trovata
        out_law_list = add_repealing_laws(law_list, self.coll_leggi_regionali, self.configs)

#        out_laws = []

        law_articles = self.find_relevant_law_articles(question_embedding, out_law_list)
#        print(law_articles[0].keys())

        return out_law_list, law_articles, entities_anno



    def find_laws_by_wording(self, question_chunks_embeddings):
        """
        Ricerca per diciture (es. "legge 15/2018", "legge regionale n. 23 del 2020").
        """

#        list_chunks = self.get_wording_similarities(question_chunks_embeddings)
        list_vector_stores = [{"index": self.leggi_regionali_wording_index, "coll_name": "leggiRegionaliWording"}]
        list_chunks = find_best_similarity(question_chunks_embeddings, list_vector_stores, 
                                           top_m = self.configs['retrievers']['law_retriever']['wording']['top_m'])

        wording_threshold = self.configs['retrievers']['law_retriever']['wording']['wording_threshold']
        
        law_ids_wording_set = set()
        law_list_wording = []

        #wording similarity
        for item in list_chunks[0:]:
            coll = item['coll']
            if item['coll'] == "leggiRegionaliWording" and item['similarity'] > wording_threshold:
                res_chunk = self.db_SAVIA_NLU[coll].find_one({"_id_faiss": int(item['_id_faiss'])})
                if res_chunk['_id_legge'] not in law_ids_wording_set:

                    law = self.coll_leggi_regionali.find_one({"_id": res_chunk['_id_legge']})
#                    print("coll:", coll, " -- sim:", item['similarity'], " -- legge:", law['legge'], " -- chunk:", res_chunk['chunk'])

                    law_list_wording.append({"similarity": item['similarity'], "_id_law": res_chunk['_id_legge'], 
                                     "chunk": res_chunk['chunk'], "pred": law['legge']})

                    law_ids_wording_set.add(res_chunk['_id_legge'])

        if self.configs['verbose']:
            print("num laws wording:", len(law_list_wording))

        laws_wording = get_law_documents(law_list_wording, self.coll_leggi_regionali, filter_repealed = False)

        return laws_wording


    def find_laws_by_reranking(self, question):
        """
        Ricerca ibrida. BM25 + re-ranker sui top_m
        """

        if self.configs['verbose']:
            print("retrieving with BM25 + re-ranking")

        ids_law_wording_list = []

        #BM25 retriever
        top_n_inds, top_n_sim = self.BM25_retriever.retrieve(question)

        reranked_items = []
        
        start_time = time.time()

        for ind_sim, ind in enumerate(top_n_inds[:]):
#            print(ind)
            out_dict = {}
            score = self.reranker.compute_score([question, self.list_all_laws[ind]['summary']], normalize=True)[0]
            law = self.list_all_laws[ind]

        #    print(ind)
            out_dict['ind'] = ind
            out_dict['_id_law'] = law['_id']
            out_dict['reranking_score'] = score
            out_dict['BM25_score'] = top_n_sim[ind_sim]
#            out_dict['gt'] = gt
            out_dict['pred'] = law['legge'].upper()

            reranked_items.append(out_dict)

        print("execution time reranking", (time.time() - start_time))
        reranked_items = sorted(reranked_items, key=lambda d: d['reranking_score'], reverse=True)

        #4.865099668502808 

        print("first scores:", [x['reranking_score'] for x in reranked_items[0:3]])

        law_list_reranking = []

        reranking_threshold = self.configs['retrievers']['law_retriever']['reranker']['reranking_threshold']

        for ind, elem in enumerate(reranked_items[0:]):
#            print(elem)
            if elem['reranking_score'] > reranking_threshold and elem['_id_law'] not in ids_law_wording_list:
#                print(elem)                
                law_list_reranking.append(elem)

        if self.configs['verbose']:
            print("num laws re-ranking:", len(law_list_reranking))

        #filtra leggi abrogate a meno che non siano in cima alla lista
        laws_reranking = get_law_documents(law_list_reranking, self.coll_leggi_regionali, filter_repealed = True)

        #seleziona un numero massimo di leggi restituite
        num_max_laws_reranking = self.configs['retrievers']['law_retriever']['reranker']['num_max_laws']
        laws_reranking = laws_reranking[0:num_max_laws_reranking]

        return laws_reranking


    def find_laws_by_semantic_similarity(self, question_embedding):
        """
        Ricerca per similarità semantica sui diversi vector store.
        """

        if self.configs['verbose']:
            print("retrieving with semantic similarity")

        list_vector_stores = [
                                {"index": self.leggi_regionali_summary_index, "coll_name": "leggiRegionaliSummary"},
                                {"index": self.leggi_regionali_HQ_on_law_index, "coll_name": "leggiRegionaliHQonLaw"},
                                {"index": self.leggi_regionali_articles_index, "coll_name": "leggiRegionaliArticles"}
                            ]

        list_chunks = find_best_similarity(question_embedding, list_vector_stores, 
                                           top_m = self.configs['retrievers']['law_retriever']['similarity']['top_m'])

        law_ids_set = set()
        law_list_semantic_similarity = []

        similarity_threshold = self.configs['retrievers']['law_retriever']['similarity']['similarity_threshold']

        #semantic similarity
        for item in list_chunks[0:]:
            coll = item['coll']
            if item['similarity'] > similarity_threshold:
                res_chunk = self.db_SAVIA_NLU[coll].find_one({"_id_faiss": int(item['_id_faiss'])})
#                res_chunk = self.coll_leggi_regionali_wording.find_one({"_id_faiss": int(item['_id_faiss'])})
                if res_chunk['_id_law'] not in law_ids_set:
#                    print("wording", item['similarity'], " --- ", res_chunk)

                    law = self.coll_leggi_regionali.find_one({"_id": res_chunk['_id_law']})
#                    print("coll:", coll, " -- sim:", item['similarity'], " -- legge:", law['legge'], " -- chunk:", res_chunk['chunk'])

                    law_list_semantic_similarity.append({"similarity": item['similarity'], "_id_law": res_chunk['_id_law'], 
                                     "chunk": res_chunk['chunk'], "pred": law['legge']})

                    law_ids_set.add(res_chunk['_id_law'])

        if self.configs['verbose']:
            print("num laws semantic similarity:", len(law_list_semantic_similarity))

        #filtra leggi abrogate a meno che non siano in cima alla lista
        laws_semantic_similarity = get_law_documents(law_list_semantic_similarity, self.coll_leggi_regionali, filter_repealed = True)

        #seleziona un numero massimo di leggi restituite
        num_max_laws_similarity = self.configs['retrievers']['law_retriever']['reranker']['num_max_laws']
        laws_semantic_similarity = laws_semantic_similarity[0:num_max_laws_similarity]

        return laws_semantic_similarity            



    def find_relevant_law_articles(self, question_embedding, out_law_list):
        """
        Estrai gli articoli delle leggi che sono importanti per 
        rispondere alla domanda dell'utente
        """

        list_vector_stores = [
                                {"index": self.leggi_regionali_articles_index, "coll_name": "leggiRegionaliArticles"},
                                {"index": self.leggi_regionali_articles_summaries_index, "coll_name": "leggiRegionaliArticlesSummaries"},
                                {"index": self.leggi_regionali_HQ_on_articles_index, "coll_name": "leggiRegionaliHQonArticles"}                                
                            ]

        list_chunks = find_best_similarity(question_embedding, list_vector_stores, 
                                           top_m = self.configs['retrievers']['law_retriever']['articles']['top_m'])

        articles_threshold = self.configs['retrievers']['law_retriever']['articles']['articles_threshold']

        articles_list = []

        id_laws = [x['_id'] for x in out_law_list]
#        print(id_laws)

#        num_max_articles_per_law = 3

        for item in list_chunks[0:]:
            coll = item['coll']
            if item['similarity'] > articles_threshold:
                res_chunk = self.db_SAVIA_NLU[coll].find_one({"_id_faiss": int(item['_id_faiss'])})
                if res_chunk['_id_law'] in id_laws:
#                    print("COLL:", coll, "SIM:", item['similarity'], " -- CHUNK:", res_chunk['chunk'], " -- ARTICLE", res_chunk['article'])
                    articles_list.append({"_id_law": res_chunk['_id_law'], "similarity": item['similarity'], 
                                        "chunk": res_chunk['chunk'], "article": res_chunk['article']})

        num_max_articles_per_law = self.configs['retrievers']['law_retriever']['articles']['num_max_articles_per_law']

#        num_max_articles_per_law = 3

        filtered_articles = self.filter_law_articles(articles_list, num_max_articles_per_law)

        if self.configs['verbose']:
            retrieved_articles = sorted(list(set([x['article'] for x in filtered_articles])))
            print("articles found:", retrieved_articles)

        return filtered_articles


    def filter_law_articles(self, articles_list, num_max_articles_per_law):
        """
        Filter up to a maximum number of relevant articles for each law
        """

        unique_laws = list(set([x['_id_law'] for x in articles_list]))
        
        filtered_articles = []

        for law in unique_laws:
            retrieved_articles = [x for x in articles_list if x['_id_law'] == law]
#            print("retrieved_articles", retrieved_articles)
#            print()
            articles = set()
            law_filtered_articles = []

            for elem in retrieved_articles:
                if elem['article'] not in articles:
                    law_filtered_articles.append(elem)
                    articles.add(elem['article'])

            law_filtered_articles = law_filtered_articles[0:num_max_articles_per_law]
            filtered_articles += law_filtered_articles


        filtered_articles = sorted(filtered_articles, key = lambda d: d['similarity'], reverse = True)

        return filtered_articles