import numpy as np
import faiss
import os 
from SAVIA_retriever.BM25.BM25_retriever import BM25Retriever
from SAVIA_retriever.utils.helper_functions import clean_title
from FlagEmbedding import FlagReranker


def find_best_similarity_on_vector_store(question_embedding, index, coll_name, top_m):
    """
    Funzione che trova la migliore similarità facendo sorting dei risultati della matrice di similarità di FAISS. 
    """

#        all_embeddings = np.concatenate([question_embedding, question_chunks_embeddings])#, question_chunks_embeddings_2])
    similarities, item_ids = index.search(question_embedding, top_m)

    similarities = similarities.flatten()
    item_ids = item_ids.flatten()

    inds = np.argsort(similarities)[::-1]

    similarities_out = similarities[inds]
    item_ids_out = item_ids[inds]

    out_list = [{"coll": coll_name, "_id_faiss": item_ids_out[ind], "similarity": x} for ind, x in enumerate(similarities_out)]

#        print(out_list)

    return out_list

def find_best_similarity(question_embedding, list_vector_stores, top_m = 1):
    """
    Funzione per ordinare i risultati della ricerca per similiarità,
    quando effettuata su vector stores diversi.
    """

    list_chunks = []

    for item in list_vector_stores:
        index = item['index']
        coll_name = item['coll_name']
        list_chunks += find_best_similarity_on_vector_store(question_embedding, index, coll_name, top_m)

    list_chunks = sorted(list_chunks, key=lambda d: d['similarity'], reverse=True)

    return list_chunks


def load_index(index_name, db, vector_store_folder, embedding_model_name):

    coll = db[index_name]
    index = faiss.read_index(os.path.join(vector_store_folder, embedding_model_name.replace("/", "_"), index_name + ".faiss"))

    return coll, index


def load_BM25_retriever(coll_leggi_regionali, top_n):
#        print("calling BM25")
    corpus = []
    list_all_laws = []

    for doc in coll_leggi_regionali.find():
#            print("here")
        titolo_orig = doc['titolo']
    #    titolo = titolo_orig
        titolo = clean_title(titolo_orig).lower()
        out_dict = {"_id": doc['_id'], "titolo": titolo, "legge": doc["legge"], 
                            "testo": doc['testo'].replace("\n", " ").lower(),#, "numero": doc["numero"],
                            "legge": doc['legge'], "numeroAnno": doc["numeroAnno"], "summary": doc['summary']}

        list_all_laws.append(out_dict)

#            corpus.append(" ".join(list(out_dict.values())))
        corpus.append(" ".join([v for k, v in out_dict.items() if k != '_id']))

#        print(corpus)

    BM25_retriever = BM25Retriever(corpus, top_n = top_n, threshold = None)

    return BM25_retriever, list_all_laws


def load_reranker(configs):

    reranker = FlagReranker(configs['models']['reranking_model'], use_fp16 = False) 

    return reranker


def postprocess_laws(laws, articles, attachments):

    list_law_keys = ["titolo", "legge", "stato", "summary", "abrogata da"]

    out_laws = []

    for law in laws:
#            print(law)
        postprocessed_law = {k: v for k, v in law.items() if k in list_law_keys}

        dict_num_atti = {"Numero atti attuativi totali": 0, 
                            "Numero atti attuativi della Giunta Regionale": 0, 
                        "Numero atti attuativi dell'Assemblea Legislativa":0
                        }

        if '_id_atti' in law.keys():
            for k, v in law['_id_atti'].items():

                if k == "attiGiuntaRegionale":
                    dict_num_atti["Numero atti attuativi della Giunta Regionale"] = len(v)
                if k == "attiAssembleaLegislativa":
                    dict_num_atti["Numero atti attuativi dell'Assemblea Legislativa"] = len(v)

            dict_num_atti['Numero atti attuativi totali'] = dict_num_atti["Numero atti attuativi della Giunta Regionale"] + dict_num_atti["Numero atti attuativi dell'Assemblea Legislativa"]

        postprocessed_law.update(dict_num_atti)

        selected_articles = list(set([x['article'] for x in articles if x['_id_law'] == law['_id']]))

#            print(selected_articles)
#            print("law id:", law['_id'])
#            print("selected articles", selected_articles)
        
        if len(selected_articles) > 0:
            law_articles = None

            if 'articoli' in law.keys(): 
                law_articles = law['articoli']

            elif 'testo_orig' in law.keys() and 'articoli' in law['testo_orig'].keys():
                law_articles = law['testo_orig']['articoli']

            for v in law_articles.keys():
#                    print(v)
                if v in selected_articles:
#                        print(articles)
                    postprocessed_law[v] = law_articles[v]

        out_laws.append(postprocessed_law)
    
    out_laws += attachments

    return out_laws