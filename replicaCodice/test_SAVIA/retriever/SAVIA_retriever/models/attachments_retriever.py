from pymongo import MongoClient
#from langchain_text_splitters import TokenTextSplitter
#import faiss
#import os
#import numpy as np
from SAVIA_retriever.models.helper_functions import find_best_similarity, load_index

class AttachmentsRetriever():
    def __init__(self, configs, client = "mongodb://mongo_db:27017", db_SAVIA = "SAVIA", db_SAVIA_NLU = "SAVIA_NLU", vector_store_folder = "../../../../../SAVIA_vector_stores"):

            self.configs = configs
            if self.configs['verbose']:
                print("loading attachments retriever")

            self.client = MongoClient(client)
            self.db_SAVIA = self.client[db_SAVIA]
            self.db_SAVIA_NLU = self.client[db_SAVIA_NLU]
            self.coll_leggi_regionali = self.db_SAVIA["leggiRegionali"]
            embedding_model_name = self.configs['models']['embedding_model']
            self.coll_leggi_regionali_attachments, self.leggi_regionali_attachments_index = load_index("leggiRegionaliAttachments", self.db_SAVIA_NLU, 
                                                                                            vector_store_folder, embedding_model_name)


    def get_attachments(self, question_chunks_embeddings, laws, entities_anno):
        """
        Funzione per il retrieval degli allegati, quali atti attuativi, link a schede tecniche e lavori preparatori.
        Se la domanda riguarda gli atti attuativi di un particolare anno, filtra la ricerca.
        """

        attachments = []

        #restituisce gli atti attuativi solo delle leggi vigenti
        active_laws = [x for x in laws if x['stato'] == "vigente"]

        list_keywords = self.get_attachments_keywords(question_chunks_embeddings)

        if self.configs['verbose'] and len(list_keywords) > 0:
            print("keywords:", list_keywords)

        num_max_atti_per_ente = self.configs['retrievers']['attachments_retriever']['num_max_atti_per_ente']

#        anno_atti = entities_anno
        if self.configs['verbose']:
            print("anno degli atti ricercati", entities_anno)


        for law in active_laws:
            titolo_legge = law['legge']

#            print(law['anno'])

            for keyword in list_keywords:
                if keyword == 'atti_attuativi':

                    list_keys_atto = ['Data adozione', 'Num.Reg.proposta / Oggetto']

                    if '_id_atti' in law.keys():
                        _id_atti = law['_id_atti']
        #                print(_id_atti)
                        for name_collection_atti, _ids_atti in _id_atti.items():
                            _ids_atti = _ids_atti[0:num_max_atti_per_ente]
#                            print(v)
#                            print("numero atti 1", len(v))
                            if name_collection_atti == "attiGiuntaRegionale":
                                coll_atti = self.db_SAVIA[name_collection_atti]

                                if len(entities_anno) > 0:
                                    atti_selected = list(coll_atti.find({"_id": {"$in": _ids_atti}, "anno": {"$in": entities_anno}}))
                                else:
                                    atti_selected = list(coll_atti.find({"_id": {"$in": _ids_atti}}))

                                if self.configs['verbose']:
                                    print("numero atti giunta trovati", len(atti_selected))

                                for atto in atti_selected:
                                    out_dict = {k: str(v).replace("\n", " ").replace("  ", " ") for k, v in atto.items() if k in list_keys_atto}
                                    out_dict['tipo atto'] = "Atto attuativo della Giunta Regionale Emilia-Romagna"
                                    out_dict['Legge Regionale di riferimento'] = titolo_legge

                                    attachments.append(out_dict)

#                                print(atti_selected)
#                                for elem in v:
            #                            print(elem)
#                                    atto = coll_atti.find_one({"_id": elem})
        #                            print("atto:", atto.keys())
#                                    list_keys_atto = ['Data adozione', 'Num.Reg.proposta / Oggetto']
#                                    out_dict = {k: str(v).replace("\n", " ").replace("  ", " ") for k, v in atto.items() if k in list_keys_atto}
#                                    out_dict['tipo atto'] = "Atto attuativo della Giunta Regionale Emilia-Romagna"
#                                    out_dict['Legge Regionale di riferimento'] = titolo_legge

#                                    attachments.append(out_dict)

                            if name_collection_atti == "attiAssembleaLegislativa":
                                coll_atti = self.db_SAVIA[name_collection_atti]
#                                dict_num_atti["Numero atti attuativi dell'Assemblea Legislativa"] = len(v)

                                if len(entities_anno) > 0:
                                    atti_selected = list(coll_atti.find({"_id": {"$in": _ids_atti}, "anno": {"$in": entities_anno}}))
                                else:
                                    atti_selected = list(coll_atti.find({"_id": {"$in": _ids_atti}}))

                                if self.configs['verbose']:
                                    print("numero atti assemblea trovati", len(atti_selected))

                                for atto in atti_selected:
                                    out_dict = {k: str(v).replace("\n", " ").replace("  ", " ") for k, v in atto.items() if k in list_keys_atto}
                                    out_dict['tipo atto'] = "Atto attuativo della Giunta Regionale Emilia-Romagna"
                                    out_dict['Legge Regionale di riferimento'] = titolo_legge

                                    attachments.append(out_dict)

#                                for elem in v:
            #                            print(elem)
#                                    atto = coll_atti.find_one({"_id": elem})
        #                            print("atto:", atto.keys())
#                                    list_keys_atto = ['Data adozione', 'Num.Reg.proposta / Oggetto']
#                                    out_dict = {k: str(v).replace("\n", " ").replace("  ", " ") for k, v in atto.items() if k in list_keys_atto}
#                                    out_dict['tipo atto'] = "Atto attuativo dell'Assemblea Legislativa Emilia-Romagna"
#                                    out_dict['Legge Regionale di riferimento'] = titolo_legge
#                                    attachments.append(out_dict)

                if keyword == 'analisi':
                    links = law['links']
                    links = [{**x, **{'Legge Regionale di riferimento': titolo_legge}} for x in links if x['tipo'] in ['Scheda tecnico-finanziaria']]

                    attachments.extend(links)

                if keyword == 'lavori_preparatori':
                    links = law['links']
    #                print(links)
                    links = [{**x, **{'Legge Regionale di riferimento': titolo_legge}} for x in links if x['tipo'] in ['Lavori preparatori']]
                    attachments.extend(links)

#                if keyword == 'circolari':
#                    print("circolari")

        return attachments
    

    def get_attachments_keywords(self, question_chunks_embeddings):
        """
        Identifica le keywords degli allegati richiesti: [atti_attuativi, analisi, lavori_preparatori, circolari].
        """

        list_vector_stores = [{"index": self.leggi_regionali_attachments_index, "coll_name": "leggiRegionaliAttachments"}]
        list_chunks = find_best_similarity(question_chunks_embeddings, list_vector_stores, top_m = 5)

        list_keywords = set()

        attachments_threshold = self.configs['retrievers']['attachments_retriever']['attachments_threshold']
#        attachments_threshold = 0.98

        for item in list_chunks:
            if item['similarity'] > attachments_threshold: 
                res_chunk = self.coll_leggi_regionali_attachments.find_one({"_id_faiss": int(item['_id_faiss'])})
#                print(item['similarity'], " ----- ", res_chunk['chunk'])
                list_keywords.add(res_chunk['key'])
        
        list_keywords = list(list_keywords)

        return list_keywords