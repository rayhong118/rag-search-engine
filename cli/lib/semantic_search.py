
from sentence_transformers import SentenceTransformer
class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')


    
    def search(self, query: str) -> list[tuple[str, float]]:
        pass


async def verify_model():
    print("Verifying model...")
    
    search_instance = SemanticSearch()
    print(f'Model loaded: {search_instance.model}')
    print(f'Max sequence length: {search_instance.model.max_seq_length}')
