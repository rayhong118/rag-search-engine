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

    # Verify modal
    verify_parser = subparsers.add_parser("verify", help="Verify the model is loaded correctly")

    # Embed
    embed_parser = subparsers.add_parser("embed_text", help="Embed text")
    embed_parser.add_argument("text", type=str, help="Text to embed")

    # Verify embeddings
    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify the embeddings")

    # Embed query
    embed_query_parser = subparsers.add_parser("embed_query", help="Embed query")
    embed_query_parser.add_argument("query", type=str, nargs="+", help="Query to embed")

    # Search
    search_parser = subparsers.add_parser("search", help="Search movie")
    search_parser.add_argument("query", type=str, nargs="+" )
    search_parser.add_argument("-l", "--limit" , type=int, default=5 )

    # Chunking. Position arg contains the content we want to chunk
    chunk_parser = subparsers.add_parser("chunk", help="Chunking")
    chunk_parser.add_argument("position", type=str, nargs="+")
    chunk_parser.add_argument("--chunk-size", type=int, default=200)
    chunk_parser.add_argument("--overlap", type=int, default=0)
    
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
        case "chunk":
            query_string = " ".join(args.position)
            chunk_size = args.chunk_size
            overlap_size = args.overlap
            words = query_string.split(" ")

            results = []
            line_count = 1
            char_count = len(query_string)

            step = max(1, chunk_size - overlap_size)
            for i in range(0, len(words), step):
                chunk = words[i:i + chunk_size]
                line_string = f"{line_count}. {' '.join(chunk)}"
                results.append(line_string)
                line_count += 1

            print(f"Chunking {char_count} characters")
            for result in results:
                print(result)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()

