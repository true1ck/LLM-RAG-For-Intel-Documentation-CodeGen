import timeit
import argparse
from flask import Flask, request, render_template
from llm.wrapper import setup_qa_chain
from llm.wrapper import query_embeddings

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        semantic_search = False
        if 'semantic_search' in request.form:
            semantic_search = True

        start = timeit.default_timer()
        if semantic_search:
            result = query_embeddings(query)
            response = result  # Just the result without additional text
            print("Query:", query)
            print("Response:", result)  # Print result directly
        else:
            qa_chain = setup_qa_chain()
            response = qa_chain({'query': query})['result']
            print("Query:", query)
            print("Response:", response)  # Print response directly
        end = timeit.default_timer()
        response_time = end - start
        
        print("Response time:", response_time)

        return render_template('index.html', query=query, response=response, response_time=response_time)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
