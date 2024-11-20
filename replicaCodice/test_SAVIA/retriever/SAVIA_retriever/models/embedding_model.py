from sentence_transformers import SentenceTransformer
#from langchain_text_splitters import TokenTextSplitter
from llama_index.core.node_parser import TokenTextSplitter

#import faiss
import numpy as np
import os


class EmbeddingModel():
    def __init__(self, configs):
        self.configs = configs
        self.model, self.embedding_dim = self.load_embedding_model()
        
    def load_embedding_model(self):

        embedding_model_name = self.configs['models']['embedding_model']
        embedding_model_path = os.path.join(self.configs['models']['models_folder'], embedding_model_name.split("/")[-1])

        if embedding_model_name.split("/")[-1] in os.listdir(self.configs['models']['models_folder']):
            if self.configs['verbose']:
                print("loading", embedding_model_name, "embedding model from:", embedding_model_path)
            embedding_model = SentenceTransformer(embedding_model_path)
        else:
            if self.configs['verbose']:
                print("loading", embedding_model_name, "embedding model from HF")
            embedding_model = SentenceTransformer(embedding_model_name, trust_remote_code=True)
            if self.configs['verbose']:
                print("saving embedding model in:", embedding_model_path)
            embedding_model.save(embedding_model_path)

        embedding_dim = embedding_model.encode(["This is a test"]).shape[1]

        return embedding_model, embedding_dim

    def create_question_embeddings(self, question):

        #embedding full question
        question = question.lower()
        question_embedding = self.model.encode(question, normalize_embeddings = True)
        #embedding question chunks
        question_chunks_embeddings = self.embed_question_chunks(question)

        return question_embedding, question_chunks_embeddings    
    

    def embed_question_chunks(self, question):
        """
        Split question in small overlapping chunks, to retrive 
        with high similarity
        """

        question = question.replace('"', "").replace('?', " ").replace('(', " ").replace(')', " ").strip()

        question_chunks = [question]

        min_chunk_size = 1
        max_chunk_size = 32

        for chunk_size in range(min_chunk_size, max_chunk_size):
                text_splitter = TokenTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_size - 1)
                chunks = text_splitter.split_text(question)
                question_chunks.extend(chunks)

        question_chunks = list(set(question_chunks))

        question_chunks_embeddings = self.model.encode(question_chunks, normalize_embeddings = True)

        return question_chunks_embeddings