from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
import logging
import time
from typing import List, Optional
import json
from functools import lru_cache

# Define a more focused prompt template
PROMPT_TEMPLATE = """
You are a factual AI assistant. Use the following context to answer the question.
If you cannot find the answer in the context, say "I cannot find the answer in the provided context."

Context:
{context}

Question: {question}

Instructions:
1. Answer based ONLY on the provided context
2. If the context doesn't contain the answer, say so
3. Keep your answer concise and factual
4. Do not add any creative or speculative information
5. If the context is unclear, say so

Answer:"""

# Set up logging
logging.basicConfig(
    filename='rag_chat.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def setup_qa_chain():
    """Set up the QA chain with the LLM and retriever"""
    # Load the vector database
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
    
    # Initialize LLM
    llm = Ollama(
        model="llama3",
        temperature=0.7  # Adjust for more/less creative responses
    )
    
    # Create prompt template
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 3}  # Limit to top 3 most relevant documents
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

def get_answer(question: str, conversation_history: Optional[List[tuple]] = None) -> str:
    """Get answer from the QA chain"""
    logging.info(f"Received question: {question}")
    if conversation_history is None:
        conversation_history = []
    
    qa_chain = setup_qa_chain()
    try:
        result = qa_chain({"query": question})
        logging.info(f"Successfully generated answer")
        # Get the answer and sources
        answer = result["result"]
        sources = result["source_documents"]
        
        # Format the response with sources
        response = f"Answer: {answer}\n\nSources:"
        for i, source in enumerate(sources, 1):
            response += f"\n{i}. {source.metadata.get('source', 'Unknown source')}"
        
        return response
    except Exception as e:
        logging.error(f"Error generating answer: {str(e)}")
        raise

def get_streaming_answer(question: str):
    qa_chain = setup_qa_chain()
    for chunk in qa_chain.stream({"query": question}):
        print(chunk, end="", flush=True)

def get_answer_with_confidence(question: str):
    result = qa_chain({"query": question})
    answer = result["result"]
    sources = result["source_documents"]
    
    # Calculate confidence based on source relevance
    confidence = sum(doc.metadata.get("score", 0) for doc in sources) / len(sources)
    
    return {
        "answer": answer,
        "confidence": confidence,
        "sources": sources
    }

def get_answer_with_retry(question: str, max_retries=3):
    for attempt in range(max_retries):
        try:
            return get_answer(question)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"Attempt {attempt + 1} failed, retrying...")
            time.sleep(1)

def get_relevant_sources(question: str, min_relevance=0.7):
    qa_chain = setup_qa_chain()
    result = qa_chain({"query": question})
    sources = result["source_documents"]
    
    # Filter sources by relevance score
    relevant_sources = [
        source for source in sources 
        if source.metadata.get("score", 0) >= min_relevance
    ]
    
    return relevant_sources

@lru_cache(maxsize=100)
def get_cached_answer(question: str):
    return get_answer(question)

def get_formatted_answer(question: str, format_type="simple"):
    answer = get_answer(question)
    
    if format_type == "simple":
        return answer
    elif format_type == "detailed":
        return f"""
Question: {question}
Answer: {answer['answer']}
Confidence: {answer['confidence']}
Sources:
{format_sources(answer['sources'])}
"""
    elif format_type == "json":
        return json.dumps(answer, indent=2)

def main():
    print("Welcome to the RAG Chatbot! Type 'quit' to exit.")
    print("Ask a question about the documents in the knowledge base.")
    
    while True:
        # Get user input
        question = input("\nYour question: ").strip()
        
        # Check if user wants to quit
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not question:
            print("Please enter a question.")
            continue
            
        try:
            answer = get_answer_with_retry(question)
            print("\n" + answer)
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()