import argparse
import json
import string
import io
from nltk.stem import PorterStemmer

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f'Searching for: {args.query}')
            keywordSearch(args.query)
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
        

if __name__ == "__main__":
    main()

