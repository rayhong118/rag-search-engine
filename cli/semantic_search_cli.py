#!/usr/bin/env python3

import json
import argparse
from lib.semantic_search import (
    verify_model, 
    embed_text, 
    verify_embeddings, 
    embed_query_text, 
    SemanticSearch,
    MOVIES_JSON_PATH
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    verify_parser = subparsers.add_parser("verify", help="Verify the model is loaded correctly")

    embed_parser = subparsers.add_parser("embed_text", help="Embed text")
    embed_parser.add_argument("text", type=str, help="Text to embed")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify the embeddings")

    embed_query_parser = subparsers.add_parser("embed_query", help="Embed query")
    embed_query_parser.add_argument("query", type=str, nargs="+", help="Query to embed")

    search_parser = subparsers.add_parser("search", help="Search movie")
    search_parser.add_argument("query", type=str, nargs="+" )
    search_parser.add_argument("-l", "--limit" , type=int, default=5 )
    
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            query_string = "".join(args.query)
            embed_query_text(query_string)
        case "search":
            query_string = " ".join(args.query)
            limit = args.limit
            with open(MOVIES_JSON_PATH, "r") as f:
                movies_list = json.load(f)["movies"]
            search_instance = SemanticSearch()
            search_instance.load_or_create_embeddings(movies_list)
            results = search_instance.search(query_string, limit)
            for index, result in enumerate(results):
                print(f"{index+1}. {result['title']} (score: {result['score']:.4f})")
                print(f"{result['description']}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()

