import argparse
import sys
import os
import json
import string
import io
from nltk.stem import PorterStemmer
import pickle

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build and save the inverted index")

    args = parser.parse_args()

    invertedIndex = InvertedIndex()

    match args.command:
        case "search":
            print(f'Searching for: {args.query}')
            try:
                invertedIndex.load()
            except Exception as e:
                print(e)
                sys.exit(1)
            keywordSearch(args.query, invertedIndex)
            pass
        case "build":
            invertedIndex.build()
            pass
        case _:
            parser.print_help()

def keywordSearch(query, index): 
    
    results = []

    movie_ids = index.get_documents(query)[:5]
    # print(movie_ids)

    for movie_id in movie_ids:
        movie_data = index.docmap[movie_id]
        results.append(movie_data)
        print(f'{movie_data["title"]} {movie_data["id"]}')

# remove stopwords
# stemming
# remove punctuation
# convert into tokens
def tokenizeStrings (input):
    result = []

    # stopwords
    stopwords = getStopWords()

    # stemming
    stemmer = PorterStemmer()

    # translation table to remove punctuation
    transTable = str.maketrans('', '', string.punctuation)

    input = input.translate(transTable)

    inputList = input.lower().split(" ")
    for inputWord in inputList:
        if inputWord != "" and inputWord not in stopwords:
            result.append(stemmer.stem(inputWord))

    return result

def getStopWords():
    with open('./data/stopwords.txt', 'r') as f:
        content = f.read()
        return content.splitlines()
    

class InvertedIndex:
    def __init__(self):
        # dictionary mapping tokens (strings) to sets of document IDs (integers)
        self.index = dict()
        # dictionary mapping document IDs to their full document objects
        self.docmap = dict()

    # Tokenize the input text, then add each token to the index with the document ID
    def  __add_document(self, doc_id, text):
        tokenizedText = tokenizeStrings(text)
        for token in tokenizedText:
            if token not in self.index:
                self.index[token] = {doc_id}
            else:
                self.index[token].add(doc_id)

    def get_documents(self, term):
        result_set = set()
        for token in tokenizeStrings(term):
            print(f"token: {token}")
            doc_ids = self.index.get(token, set())
            result_set.update(doc_ids)
        return sorted(list(result_set))

    def build(self):
        with open('./data/movies.json', 'r') as f:
            data = json.load(f)
            movies = data["movies"]

        for movie in movies:
            movie_info = f'{movie["title"]} {movie["description"]}'
            self.__add_document(movie["id"], movie_info)
            self.docmap[movie["id"]] = movie

        self.save()

    def save(self):
        os.makedirs("cache", exist_ok=True)
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)

    def load(self):
        try:
            with open("cache/index.pkl", "rb") as f:
                self.index = pickle.load(f)
            with open("cache/docmap.pkl", "rb") as f:
                self.docmap = pickle.load(f)
        except FileNotFoundError:
            raise Exception("Search index not found. Please run the 'build' command first.")

if __name__ == "__main__":
    main()