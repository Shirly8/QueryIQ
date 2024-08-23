import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from .RAG import initialize_chroma
import re

#To run: python3 QUERY.py --query "What time does Aretti open?"

CHROMA_PATH = "./VectorDB"

def query_rag(query_text):

    #Initialize Database
    db = initialize_chroma()

    #Search database
    results = db.similarity_search_with_score(query_text, k=5)

    #Initialize Ollama to get the answer
    llm = Ollama(model = "llama3")

    #combine all the chunks and pass it to Ollama
    all_context = "\n".join([doc.page_content for doc, score in results])

    #Create a prompt for Ollama
    prompt_template = ChatPromptTemplate.from_template(
        f"You are an intelligent customer service agent. Provide the best customer service answering to the queries\n" 
        f"This is the query: {query_text}.\n"
        f"This is the context: {all_context}"
    )

    prompt = prompt_template.format(context=all_context, query_text=query_text)

    # Run the query using the Ollama model
    response = llm.generate([prompt])

    #Extract and print the resposne text
    if response and response.generations:
        generation_chunk = response.generations[0][0]
        response_text = generation_chunk.text

        print("Query Response: ", response_text)
        return 
    
    return

def main():
    parser = argparse.ArgumentParser(description="Process query with RAG")
    parser.add_argument("--query", type=str, required=True, help="The query text")
    args = parser.parse_args()
    query_rag(args.query)

if __name__ == "__main__":
    main()