from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
#from bson.objectid import ObjectId
import faiss
import os
from langchain_text_splitters import TokenTextSplitter
import numpy as np
from gliner import GLiNER

import sys
sys.path.append("../../../")
sys.path.append("../../")

from utils_retriever.helper_functions import load_configs
#from BM25.BM25_retriever import BM25Retriever
#from llama_index.core.node_parser import SentenceSplitter
import random
from models.law_retriever import LawRetriever
from models.NER_model import NERModel
from models.embedding_model import EmbeddingModel


class SAVIARetriever():
    """
    Classe che implementa il retriever. Il sistema è modulare,
    un primo modulo identifica la legge in base alla similarità con la dicitura,
    es. "legge regionale 15/2018". Se la legge non è espicitamente presente nella domanda, 
    effetta la ricerca per similarità sui chunk di titolo, argomenti e primi due articoli,
    o sul summary.
    Un secondo modulo identifica quali documenti vanno aggiunti al prompt (es. atti attuativi,
    lavori preparatori, documenti tecnici). Un ulteriore modulo di NER identifica le entities 
    utili, ad esempio se l'utente vuole limitare la ricerca agli atti/documenti di un particolare 
    anno (o più anni). Questa parte verra sostituita da un sistema generativo.
    """
    def __init__(self, configs_path="../configs/configs.yml", embedding_model_name = "BAAI/bge-m3", client = "mongodb://mongo_db:27017", db = "SAVIA", coll = "leggiRegionali",
                vector_store_folder = "../../../../SAVIA_vector_stores"):
        self.configs = load_configs(configs_path = configs_path)

#        self.NER_model = NERModel()
        self.embedding_model = EmbeddingModel(self.configs)
        self.law_retriever = LawRetriever(self.configs)

#        print(self.configs)

        """
        self.embedding_model, self.embedding_dim = self.load_embedding_model(embedding_model_name)
        self.NER_model, self.NER_labels = self.load_NER_model()
        self.client = MongoClient(client)
        self.db = self.client[db]
        self.coll_leggi_regionali = self.db[coll]
        self.vector_store_folder = vector_store_folder
#        self.BM25_retriever, self._id_leggi = self.load_BM25_retriever(top_n = 30)
#        self.leggi_regionali_diciture_index = self.load_diciture_index()
#        self.leggi_regionali_articles_index = self.load_articles_index()
        self.leggi_regionali_allegati_index = self.load_allegati_index()

        """

    def create_prompt(self, question):

#        entities = self.NER_model.extract_entities(question)

        question_embedding, question_chunks_embeddings, question_chunks_embeddings_2 = self.embedding_model.create_question_embeddings(question)

        law_list = self.law_retriever.get_law(question_embedding, question_chunks_embeddings, question_chunks_embeddings_2) 

#        print(law_list)

#        print(question_embedding.shape)

        """
        entities = self.extract_entities(question)
#        print(entities)

        for entity in entities:
            print(entity["text"], "=>", entity["label"])

#        dict_chunks = self.retrieve_by_similarity(question)

#        law_list = self.get_law(dict_chunks)
#        print("anno legge:", law['anno'])
#        print(law_list)
#        print(law_list[0]['legge'])

        entities_anni = [int(x['text']) for x in entities if x['text'] != law_list[0]['anno']]
        print("entities anni:", entities_anni)

        prompt = []

        if len(law_list) > 0:
            related_docs, numero_atti, numero_atti_GR, numero_atti_AL = self.get_related_docs(dict_chunks, law_list[-1], entities_anni)

            list_keys = ["legge", "titolo", "anno", "data", "numero", "stato", "legge abrogante"]


            for law in law_list:
                out_law = {k: v for k, v in law.items() if k in list_keys}
                
                if law['stato'] == 'vigente':
                    out_law['numero atti attuativi'] = numero_atti

                prompt.append(out_law)

    #        out_law['numero atti attuativi'] = numero_atti
    #        out_law['numero atti attuativi Giunta Regionale'] = numero_atti_GR
    #        out_law['numero atti attuativi Assemblea Legislativa'] = numero_atti_AL

    #        print(out_law)
    #        print()
    #        related_docs = related_docs[0:10]

    #        prompt.append(out_law)
            prompt.extend(related_docs)
#        print(dict_chunks)
        """

        prompt = None

        return law_list


    """
    def load_BM25_retriever(self, top_n):

        list_all_laws = []
        corpus = []
        _id_leggi = []

        for ind, doc in enumerate(self.coll_leggi_regionali.find()[0:]):
            titolo_orig = doc['titolo']
        #    titolo = titolo_orig
            titolo = clean_title(titolo_orig).lower()
            out_dict = {"titolo": titolo, "legge": doc["legge"], 
                                "testo": doc['testo'].replace("\n", " ").lower(),#, "numero": doc["numero"],
                                "legge": doc['legge']}#, "numeroAnno": doc["numeroAnno"]}

            if 'articoli' in doc.keys():
                articoli = list(doc['articoli'].values())
                articoli_str = " ".join([x for x in articoli[0:2]]).lower()
                
                out_dict["articoli"] = articoli_str

            list_all_laws.append(out_dict)

            corpus.append(" ".join(list(out_dict.values())))

            _id_leggi.append(doc['_id'])

        retriever = BM25Retriever(corpus, top_n = top_n, threshold = None)


        return retriever, _id_leggi
    """

#    def load_diciture_index(self):
#        
#        index_name = "leggiRegionaliDiciture"
#        self.coll_leggi_regionali_diciture = self.db[index_name]#

#        leggi_regionali_diciture_index = faiss.read_index(os.path.join(self.vector_store_folder , index_name + ".faiss"))

#        return leggi_regionali_diciture_index
    
#    def load_articles_index(self):
        
#        index_name = "leggiRegionaliArticles"
#        self.coll_leggi_regionali_articles = self.db[index_name]

#        leggi_regionali_articles_index = faiss.read_index(os.path.join(self.vector_store_folder , index_name + ".faiss"))

#        return leggi_regionali_articles_index

    def load_allegati_index(self):
        
        index_name = "leggiRegionaliAllegati"
        self.coll_leggi_regionali_allegati = self.db[index_name]

        leggi_regionali_allegati_index = faiss.read_index(os.path.join(self.vector_store_folder , index_name + ".faiss"))

        return leggi_regionali_allegati_index

    """
    def retrieve_with_diciture(self, question_chunks_embeddings, top_m = 1):
        
#        top_m = 1
#        retrieved_item = None
#        print(question_chunks)

        similarity, item_ids = self.leggi_regionali_diciture_index.search(question_chunks_embeddings, top_m)
#        similarity = similarity[0]
#        item_ids = item_ids[0]

        similarity = similarity[:, 0]
        item_ids = item_ids[:, 0]

        max_similarity_id = np.argmax(similarity)
        best_keyword_similarity = similarity[max_similarity_id]
        best_keyword_id = item_ids[max_similarity_id]

        return best_keyword_id, best_keyword_similarity


    def retrieve_with_articles(self, question_embedding, top_m = 5):
        
#        retrieved_item = None
#        print(question_chunks)

        similarity, item_ids = self.leggi_regionali_articles_index.search(question_embedding, top_m)
        best_article_similarity = similarity[0]
        best_article_ids = item_ids[0]

        return best_article_ids, best_article_similarity
    """

    def retrieve_related_docs(self, question_chunks_embeddings, top_m = 1):
        
#        top_m = 5
#        retrieved_item = None

        similarity, item_ids = self.leggi_regionali_allegati_index.search(question_chunks_embeddings, top_m)
#        print(similarity)

        similarity = similarity[:, 0]
        item_ids = item_ids[:, 0]
#        print()
#        print(similarity)
#        print()
#        print(item_ids)

        #sort by best similarity
        sorted_similarity_ind = list(np.argsort(similarity)[::-1])

        sorted_similarity = [similarity[ind] for ind in sorted_similarity_ind] 
        sorted_item_ids = [item_ids[ind] for ind in sorted_similarity_ind] 

#        print(sorted_similarity)
#        print(sorted_item_ids)
#        similarity = list(similarity)

        related_docs_similarity = sorted_similarity
        related_docs_ids = sorted_item_ids

#        print()
#        print(res) 
#        print(related_docs_similarity[res[0]])

#        similarity = similarity[:, 0]
#        item_ids = item_ids[:, 0]

#        max_similarity_id = np.argmax(similarity)
#        best_similarity = similarity[max_similarity_id]
#        best_item_id = item_ids[max_similarity_id]

#        print(best_similarity)#, item_ids)
#        key_allegato = None

#        if best_similarity > threshold:
#            print("trovato allegato")
#            res_chunk = self.coll_leggi_regionali_allegati.find_one({"_id_faiss": int(best_item_id)})
#            print(res_chunk,  "sim:", best_similarity)
#            key_allegato = res_chunk['key_allegato']
#            _id_legge = res_chunk['_id_legge']


#        similarity = similarity[:, 0]
#        item_ids = item_ids[:, 0]

#        max_similarity_id = np.argmax(similarity)
#        best_similarity = similarity[max_similarity_id]
#        best_item_id = item_ids[max_similarity_id]

#        print(best_similarity)
#        print(best_item_id)

#        res_chunk = self.coll_leggi_regionali_allegati.find_one({"_id_faiss": int(best_item_id)})
#        print(res_chunk)
#        _id_legge = res_chunk['_id_legge']
#        res_legge = self.coll_leggi_regionali.find_one({"_id": _id_legge})
#        legge = res_legge['legge']
#        print("legge:", legge, "- sim:", round(best_similarity, 3))

#        if best_similarity > threshold:
#            retrieved_item = legge

        return related_docs_ids, related_docs_similarity

    """
    def retrieve_by_similarity(self, question):

        dict_chunks = {}

        #embedding question
        question = question.lower()
        question_embedding = self.embedding_model.encode([question])
        faiss.normalize_L2(question_embedding)

        #embedding question chunks
#        question_chunks = []
        text_splitter = TokenTextSplitter(chunk_size = 30, chunk_overlap = 25)
        question_chunks = text_splitter.split_text(question)
        question_chunks.append(question)
        question_chunks_embeddings = self.embedding_model.encode(question_chunks)
        faiss.normalize_L2(question_chunks_embeddings)


#        question_chunks_2 = []
        text_splitter = TokenTextSplitter(chunk_size = 10, chunk_overlap = 5)
        question_chunks_2 = text_splitter.split_text(question)
#        question_chunks_2.append(question)
#        print(question_chunks_2)
        question_chunks_embeddings_2 = self.embedding_model.encode(question_chunks_2)
        faiss.normalize_L2(question_chunks_embeddings_2)


#        print("question chunks:", question_chunks)
#        print("question chunks 2:", question_chunks_2)

        best_keyword_id, best_keyword_similarity = self.retrieve_with_diciture(question_chunks_embeddings)
#        best_keyword_id, best_keyword_similarity = self.retrieve_with_keywords(question_chunks_embeddings_2)

#        print("result keywords search:", retrieved_item_keyword)

        dict_chunks['best_item_keyword'] = [{"_id_faiss": best_keyword_id, "similarity": best_keyword_similarity}]

#        if retrieved_item is None:
#        if False:
        best_article_ids, best_article_similarity = self.retrieve_with_articles(question_embedding)
#            print("result articles search:", retrieved_item_chunk)
#            retrieved_item = retrieved_item_chunk 
        
#        dict_chunks['retrieved_item_chunk'] = {"_id_faiss": best_article_ids, "similarity": best_article_similarity}

        dict_chunks['retrieved_item_chunk'] = [{"_id_faiss": x, "similarity": best_article_similarity[ind]} 
                                               for ind, x in enumerate(best_article_ids)]

        related_docs_ids, related_docs_similarity = self.retrieve_related_docs(question_chunks_embeddings_2)
#        print("key allegato:", key_allegato)

#        dict_chunks['related_docs'] = {"_id_faiss": related_docs_ids, "similarity": related_docs_similarity}

        dict_chunks['related_docs'] = [{"_id_faiss": x, "similarity": related_docs_similarity[ind]} 
                                               for ind, x in enumerate(related_docs_ids)]


#        print(dict_chunks)

        return dict_chunks
    """


    def get_related_docs(self, dict_chunks, law, entities_anni):
#        print(law)

#        print(dict_chunks['related_docs'])

        threshold = 0.7

        dict_related_docs = [x for x in dict_chunks['related_docs'] if x['similarity'] > threshold]
#        print()
#        print(dict_related_docs)

        _ids_faiss = list(set([x['_id_faiss'] for x in dict_related_docs]))
#        print(_ids_faiss)

        out_list = []

        related_docs_keys = []

        for _id_faiss in _ids_faiss[0:]:
#            _id_faiss =  item['_id_faiss']
#            print(_id_faiss)

            res_chunk = self.coll_leggi_regionali_allegati.find_one({"_id_faiss": int(_id_faiss)})
            related_docs_keys.append(res_chunk['key_allegato'])

        related_docs_keys = list(set(related_docs_keys))

#        print(related_docs_keys)

        numero_atti = 0
        numero_atti_GR = 0
        numero_atti_AL = 0

        for related_docs_key in related_docs_keys[0:]:

#            print(related_docs_key)
#            print(law.keys())
#            print(law['_id_atti'])

            #funzione buona, per quando ci sarà di nuovo il match tra atti e leggi
            """
            if related_docs_key == 'atti_attuativi':
                if '_id_atti' in law.keys():
                    _id_atti = law['_id_atti']
    #                print(_id_atti)
                    for k, v in _id_atti.items():
    #                    print(k, v)

                        if k == "attiGiuntaRegionale":
                            coll_atti = self.db[k]
                            for elem in v[0:5]:
    #                            print(elem)
                                atto = coll_atti.find_one({"_id": elem})
    #                            print("atto:", atto.keys())
                                list_keys_atto = ['anno', 'Tipo Atto', 'Num adozione', 'Data adozione', 'Num.Reg.proposta / Oggetto']
                                out_dict = {k: str(v).replace("\n", " ").replace("  ", " ") for k, v in atto.items() if k in list_keys_atto}
                                out_dict['ente'] = "Giunta Regionale"

                                out_list.append(out_dict)
            """

            if related_docs_key == 'atti_attuativi' or related_docs_key == "atti_attuativi_GR" or related_docs_key == "atti_attuativi_AL":
#                colls = ["attiGiuntaRegionale", "attiAssembleaLegislativa"]

#                print(colls)
                    #funzione dummy
        #            if related_docs_key == 'atti_attuativi':
#                for coll in colls[0:1]:
#                    print(coll)
#                res = self.coll_leggi_regionali.find_one({'_id': law['_id']})

                if '_id_atti' in law.keys():
                    _id_atti = law['_id_atti']

#                    print(_id_atti)

                    for k, v in _id_atti.items():
#                        print(k, v) 
                        coll_atti = self.db[k]
#                        print("values", v)
                        if len(entities_anni) > 0:
                            atti_selected = list(coll_atti.find({"_id": {"$in": v}, "anno": {"$in": entities_anni}}))
                        else:
                            atti_selected = list(coll_atti.find({"_id": {"$in": v}}))

#                        print(len(atti_selected))

                        for atto in atti_selected[0:10]:
#                            print(atto['_id'])
                            out_dict = {}
                            out_dict['legge di riferimento'] = law['legge']
                            list_keys_atto = ['anno', 'Tipo Atto', 'Num adozione', 'Data adozione', 'Num.Reg.proposta / Oggetto']
#                            print("here", law['legge'])
                            if k == "attiGiuntaRegionale":
                                out_dict['ente'] = "Giunta Regionale"
                                numero_atti_GR += 1

                            else:
                                out_dict['ente'] = "Assemblea Legislativa"
                                numero_atti_AL += 1

                            out_dict.update({k: str(v).replace("\n", " ").replace("  ", " ") for k, v in atto.items() if k in list_keys_atto})
                            out_list.append(out_dict)
                            
#                            numero_atti += 1

#                    coll_atti = self.db["attiGiuntaRegionale"]
#                    atti = list(coll_atti.find({})[0:200])

#                    num_atti = random.randint(0, 5)

#                atti_selected = random.sample(atti, num_atti)
#                for atto in atti_selected:
#                    list_keys_atto = ['anno', 'Tipo Atto', 'Num adozione', 'Data adozione', 'Num.Reg.proposta / Oggetto']
#                    out_dict = {k: str(v).replace("\n", " ").replace("  ", " ") for k, v in atto.items() if k in list_keys_atto}
#                    out_dict['ente'] = "Giunta Regionale"#
#                    out_list.append(out_dict)

            if related_docs_key == 'analisi':
#                print(law.keys())
#                print(law['links'])
#                print("Here")
                links = law['links']
#                print(links)
                links = [x for x in links if x['tipo'] in ['Scheda tecnico-finanziaria']]
                out_list.extend(links)


            if related_docs_key == 'lavori_preparatori':
#                print(law.keys())
#                print(law['links'])
                links = law['links']
#                print(links)
                links = [x for x in links if x['tipo'] in ['Lavori preparatori']]
                out_list.extend(links)

#                print(links)
#        print()
#        print(out_list)

#        print("Numero atti:", numero_atti)
#        print("Numero atti GR:", numero_atti_GR)
#        print("Numero atti AL:", numero_atti_AL)

#        out_list = list(set(out_list))

        # using frozenset to
        # remove duplicates
        out_list = list({frozenset(item.items()) : 
                    item for item in out_list}.values())


        numero_atti += len(out_list)
#        print(res_list)

        return out_list, numero_atti, numero_atti_GR, numero_atti_AL
