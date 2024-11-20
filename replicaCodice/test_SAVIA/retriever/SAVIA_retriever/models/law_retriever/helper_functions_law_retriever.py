#import numpy as np
#import faiss
import os 
from SAVIA_retriever.BM25.BM25_retriever import BM25Retriever
from SAVIA_retriever.utils.helper_functions import clean_title
from FlagEmbedding import FlagReranker


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


def get_law_documents(law_list, coll_leggi_regionali, filter_repealed = True):
    """
    Funzione per filtrare le leggi abrogate. 
    Restitusce leggi abrogate solo se si trovano al primo posto della ricerca per similarità.
    Questo per evitare di intasare il contesto fornendo troppe informazioni.     
    """

    law_documents = []
#        print("orig", law_list)

    for ind, elem in enumerate(law_list):
        target_law = coll_leggi_regionali.find_one({"_id": elem['_id_law']})
#            print("LAW", law)
        if filter_repealed:
            if ind == 0:
                law_documents.append(target_law)

            elif ind > 0 and target_law['stato'] == 'vigente':
                law_documents.append(target_law)
        else:
            law_documents.append(target_law)

    return law_documents


def add_repealing_laws(law_list, coll_leggi_regionali, configs):
    """
    Funzione per aggiungere la legge abrogante nel caso una legge sia abrogata.
    Inolte aggiunge alle leggi regionali alcuni campi utili per il RAG (es. tipo atto).
    """

    out_law_list = []

    law_ids = [x['_id'] for x in law_list]

    
    for law in law_list:
        law['tipo atto'] = "Legge Regionale dell'Emilia-Romagna"

        legge_abrogante = None

        if law['stato'] != "vigente" and '_id_legge_abrogante' in law.keys():
            if configs['verbose']:
                print("legge:", law['legge'], " --- ABROGATA")

            legge_abrogante = coll_leggi_regionali.find_one({"_id": law['_id_legge_abrogante']})
            legge_abrogante['tipo atto'] = "Legge Regionale dell'Emilia-Romagna"

#                print(legge_abrogante['urn'])
            law['abrogata da'] = legge_abrogante['legge']

        out_law_list.append(law)

#            print(legge_abrogante.keys())
        if legge_abrogante and legge_abrogante['_id'] not in law_ids:
            out_law_list.append(legge_abrogante)


    return out_law_list
