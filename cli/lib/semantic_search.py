import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MOVIES_JSON_PATH = os.path.join(BASE_DIR, "data", "movies.json")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")
class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.embeddings: np.ndarray | None = None
        self.documents: list[str] | None = None
        self.document_map= dict()
    
    def search(self, query: str) -> list[tuple[str, float]]:
        pass

    def generate_embedding(self, text: str):
        if len(text) == 0 or text.strip() == "":
            raise ValueError("Text cannot be empty")

        embedding = self.model.encode([text])[0]
        return embedding

    def build_embedding(self, documents: list[dict[str, str]]):
        if len(documents) == 0:
            raise ValueError("Documents cannot be empty")

        self.documents = documents

        document_embeddings = []
        document_strings = []
        for document in documents:
            self.document_map[document['id']] = document
            document_string = f"{document['title']}: {document['description']}"
            document_strings.append(document_string)

        self.embeddings = self.model.encode(document_strings, show_progress_bar=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for document in documents:
            self.document_map[document['id']] = document
        
        if os.path.exists(EMBEDDINGS_PATH):
            self.embeddings = np.load(EMBEDDINGS_PATH)

            if len(self.embeddings) != len(documents):
                self.build_embedding(documents)
            else:
                return self.embeddings
        else:
            self.build_embedding(documents)
        
        

def verify_model():
    print("Verifying model...")
    
    search_instance = SemanticSearch()
    print(f'Model loaded: {search_instance.model}')
    print(f'Max sequence length: {search_instance.model.max_seq_length}')

def embed_text(text:str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    search_instance = SemanticSearch()
    with open(MOVIES_JSON_PATH, 'r') as f:
        movies_list = json.load(f)
    embeddings = search_instance.load_or_create_embeddings(movies_list)
    print(f"Number of docs:   {len(movies_list)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
