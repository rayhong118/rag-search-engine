import argparse
import json

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

def keywordSearch(keyword): 
    with open('./data/movies.json', 'r') as f:
        dictionary = json.load(f)

    movies = dictionary["movies"]

    results = []

    for movie in movies:
        title = movie["title"]
        if keyword.lower() in title.lower():
            results.append(movie)
            if len(results) == 5:
                break

    for i, result in enumerate(results):
        print(f'{i+1}. {result["title"]}')
    
    pass


if __name__ == "__main__":
    main()