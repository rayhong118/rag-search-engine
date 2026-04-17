import argparse
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
            keywordSearch(args.query)
            pass
        case "build":
            invertedIndex.build()
            docs = invertedIndex.get_documents("merida")
            print(f"First document for token 'merida' = {docs[0]}")
            pass
        case _:
            parser.print_help()

def keywordSearch(query): 
    with open('./data/movies.json', 'r') as f:
        dictionary = json.load(f)

    movies = dictionary["movies"]

    results = []

    tokenizedQuery = tokenizeStrings(query)

    print(tokenizedQuery)

    # translation table to remove punctuation
    transTable = str.maketrans('', '', string.punctuation)

    for movie in movies:

        title = movie["title"].translate(transTable)
        tokenizedTitle = tokenizeStrings(title)

        for query_word in tokenizedQuery:
            if any(query_word in title_word for title_word in tokenizedTitle):
                results.append(movie)
                break


    for i, result in enumerate(results):
        print(f'{i+1}. {result["title"]}')
    
    pass

def tokenizeStrings (input):
    result = []

    # stopwords
    stopwords = getStopWords()

    # stemming
    stemmer = PorterStemmer()

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

if __name__ == "__main__":
    main()