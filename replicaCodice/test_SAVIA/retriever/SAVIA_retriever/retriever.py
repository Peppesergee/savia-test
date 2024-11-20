#from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
#from bson.objectid import ObjectId
#import faiss
import os
#from langchain_text_splitters import TokenTextSplitter
import numpy as np
#from gliner import GLiNER

#import sys
#sys.path.append("../../../")
#sys.path.append("../../")
#sys.path.append("../")

from SAVIA_retriever.utils_retriever.helper_functions import load_configs
#from BM25.BM25_retriever import BM25Retriever
#from llama_index.core.node_parser import SentenceSplitter
#import random
#from models.law_retriever import LawRetriever
#from models.law_retriever_test import LawRetriever
#from models.law_retriever_BM25_reranking import LawRetriever
from SAVIA_retriever.models.law_retriever.law_retriever import LawRetriever

from SAVIA_retriever.models.NER_model import NERModel
from SAVIA_retriever.models.embedding_model import EmbeddingModel
from SAVIA_retriever.models.attachments_retriever import AttachmentsRetriever
#from models.articles_retriever import ArticlesRetriever
from SAVIA_retriever.models.law_retriever.helper_functions_law_retriever import postprocess_laws

class SAVIARetriever():
    def __init__(self, configs_path="./SAVIA_retriever/configs/configs.yml",
                vector_store_folder = "/SAVIA_vector_stores"):
        self.configs = load_configs(configs_path = configs_path)
        
        self.embedding_model = EmbeddingModel(self.configs)
        self.NER_model = NERModel(self.configs)
        self.law_retriever = LawRetriever(self.configs, vector_store_folder = vector_store_folder)
        self.attachments_retriever = AttachmentsRetriever(self.configs, vector_store_folder = vector_store_folder)


    def create_prompt(self, question):

        _, entities_anno = self.NER_model.extract_entities(question)

        question_embedding, question_chunks_embeddings = self.embedding_model.create_question_embeddings(question)
        question_embedding = np.expand_dims(question_embedding, axis=0)

#        entities = self.NER_model.extract_entities(question)
        laws, articles, entities_anno = self.law_retriever.get_laws(question, question_embedding, question_chunks_embeddings, entities_anno)

        attachments = self.attachments_retriever.get_attachments(question_chunks_embeddings, laws, entities_anno)

        out_list = postprocess_laws(laws, articles, attachments)

        if self.configs['verbose']:
            print("num retrieved items:", len(out_list))

        return out_list
