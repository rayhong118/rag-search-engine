import argparse
import sys
import os
import json
import string
import io
from nltk.stem import PorterStemmer
import pickle
from collections import defaultdict, Counter
import math

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build and save the inverted index")
    
    term_frequencies_parser = subparsers.add_parser("tf", help="Term frequency")
    term_frequencies_parser.add_argument("documentID", type=int, help="Document ID")
    term_frequencies_parser.add_argument("term", type=str, help="Term")

    idf_parser = subparsers.add_parser('idf')
    idf_parser.add_argument("term", type=str)

    tfidf_parser = subparsers.add_parser('tfidf')
    tfidf_parser.add_argument("doc_id", type=int)
    tfidf_parser.add_argument("term", type=str)

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
        case "tf":
            try:
                invertedIndex.load()
                count = invertedIndex.get_tf(args.documentID, args.term)
                print(f"Frequency of '{args.term}' in document {args.documentID}: {count}")
            except Exception as e:
                print(e)
                sys.exit(1)
        case "idf":
            invertedIndex.load()
            idf = calculateIDF(invertedIndex.get_documents(args.term), invertedIndex.docmap)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
            pass
        case "tfidf":
            invertedIndex.load()
            tf = invertedIndex.get_tf(args.doc_id, args.term)
            idf = calculateIDF(invertedIndex.get_documents(args.term), invertedIndex.docmap)
            tf_idf = tf * idf
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
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

    inputList = input.lower().split()
    for inputWord in inputList:
        if inputWord != "" and inputWord not in stopwords:
            result.append(stemmer.stem(inputWord))

    return result

def getStopWords():
    with open('./data/stopwords.txt', 'r') as f:
        content = f.read()
        return content.splitlines()

def calculateIDF(doc_ids, docmap):
    total_doc_count = len(docmap)
    term_match_doc_count = len(doc_ids)
    print(f'total_doc_count: {total_doc_count}, term_match_doc_count: {term_match_doc_count}')
    return math.log((total_doc_count + 1) / (term_match_doc_count + 1))
    
class InvertedIndex:
    def __init__(self):
        """Initializes the inverted index, document mapping, and term frequency storage."""
        # mapping tokens (strings) to sets of document IDs (integers)
        self.index = dict()
        # mapping document IDs to their full document objects
        self.docmap = dict()
        # nested mapping: {doc_id: {token: frequency}}
        self.term_frequencies = defaultdict(Counter)

    def  __add_document(self, doc_id, text):
        """Internal method to tokenize text and update the index and term frequencies for a document."""
        tokenizedText = tokenizeStrings(text)
        for token in tokenizedText:
            if token not in self.index:
                self.index[token] = {doc_id}
            else:
                self.index[token].add(doc_id)
            
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term):
        """Returns a sorted list of unique document IDs that contain any tokens from the search term."""
        result_set = set()
        for token in tokenizeStrings(term):
            print(f"token: {token}")
            doc_ids = self.index.get(token, set())
            result_set.update(doc_ids)
        return sorted(list(result_set))

    def get_tf(self, doc_id, term):
        """Retrieves the frequency of a specific term within a given document ID."""
        tokens = tokenizeStrings(term)
        if not tokens:
            return 0
        token = tokens[0]
        return self.term_frequencies.get(doc_id, {}).get(token, 0)

    def build(self):
        """Loads movies from the JSON file and builds the complete inverted index and docmap."""
        with open('./data/movies.json', 'r') as f:
            data = json.load(f)
            movies = data["movies"]

        for movie in movies:
            movie_info = f'{movie["title"]} {movie["description"]}'
            self.__add_document(movie["id"], movie_info)
            self.docmap[movie["id"]] = movie

        self.save()

    def save(self):
        """Pickles and saves the index, docmap, and term frequencies to the cache directory."""
        os.makedirs("cache", exist_ok=True)
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)
            
    def load(self):
        """Loads the pickled index, docmap, and term frequencies from the cache directory."""
        try:
            with open("cache/index.pkl", "rb") as f:
                self.index = pickle.load(f)
            with open("cache/docmap.pkl", "rb") as f:
                self.docmap = pickle.load(f)
            with open("cache/term_frequencies.pkl", "rb") as f:
                self.term_frequencies = pickle.load(f)
        except FileNotFoundError:
            raise Exception("Search index not found. Please run the 'build' command first.")

if __name__ == "__main__":
    main()