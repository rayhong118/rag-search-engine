
from sentence_transformers import SentenceTransformer
class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')


    
    def search(self, query: str) -> list[tuple[str, float]]:
        pass

    def generate_embedding(self, text: str):
        if len(text) == 0 or text.strip() == "":
            raise ValueError("Text cannot be empty")

        embedding = self.model.encode([text])[0]
        return embedding


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