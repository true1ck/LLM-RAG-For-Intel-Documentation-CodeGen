import argparse
import timeit
import torch
from llm.wrapper import setup_qa_chain
from llm.wrapper import query_embeddings

# Function to initialize model on GPU
def initialize_model_on_gpu():
    # Initialize your model here, ensuring it's placed on the GPU
    model = setup_qa_chain()
    return model

# Function to process query on GPU
def process_query_on_gpu(query, model, semantic_search):
    if semantic_search:
        # Run semantic search on GPU
        result = query_embeddings(query)
    else:
        # Run question answering on GPU
        result = model({'query': query})
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        type=str,
                        default='What is the invoice number value?',
                        help='Enter the query to pass into the LLM')
    parser.add_argument('--semantic_search',
                        action='store_true',
                        help='Enter True if you want to run semantic search, else False')
    args = parser.parse_args()

    # Initialize model on GPU
    model = initialize_model_on_gpu()

    queries = [args.input] * 10  # Example: 10 copies of the same query for demonstration

    start = timeit.default_timer()

    results = []
    for query in queries:
        result = process_query_on_gpu(query, model, args.semantic_search)
        results.append(result)

    for idx, result in enumerate(results):
        if args.semantic_search:
            print(f'Query {idx + 1} - Semantic search: {result}')
        else:
            print(f'Query {idx + 1} - Answer: {result["result"]}')

    end = timeit.default_timer()

    print(f"Time to retrieve answers: {end - start}")
