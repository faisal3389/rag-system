from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
import glob

# Create persist directory if it doesn't exist
persist_directory = "chroma_db"
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

# Load all text files in data/dev_notes
print("Loading documents...")
file_paths = glob.glob("data/dev_notes/*")
docs = []
for file_path in file_paths:
    try:
        loader = TextLoader(file_path)
        docs.extend(loader.load())
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
print(f"Loaded {len(docs)} documents from {len(file_paths)} files")

# Split into chunks
print("Splitting documents into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks")

# Embed and store
print("Creating embeddings and storing in Chroma...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory
)
print(f"Documents stored in {persist_directory}")

# Verify the database
print("\nVerifying database...")
print(f"Number of documents in database: {db._collection.count()}")
print("Database verification complete!")

# Define the prompt template
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}

Question: {question}

Answer the question based on the above context. If you cannot find the answer in the context, say "I cannot find the answer in the provided context."
"""

# Initialize the LLM and QA chain
def setup_qa_chain():
    """Set up the QA chain with the LLM and retriever"""
    llm = Ollama(model="llama3")
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

def query_rag(question: str):
    """
    Query the RAG system and return formatted response with sources
    
    Args:
        question (str): The question to ask
        
    Returns:
        tuple: (formatted_response, response_text, sources)
    """
    qa_chain = setup_qa_chain()
    result = qa_chain({"query": question})
    
    # Extract response and sources
    response_text = result["result"]
    source_docs = result["source_documents"]
    
    # Format sources
    sources = []
    for doc in source_docs:
        source = {
            "content": doc.page_content[:200] + "...",  # First 200 chars
            "metadata": doc.metadata
        }
        sources.append(source)
    
    # Format the complete response
    formatted_response = f"""
Question: {question}

Answer: {response_text}

Sources:
{'-' * 50}
"""
    for i, source in enumerate(sources, 1):
        formatted_response += f"""
Source {i}:
Content: {source['content']}
Metadata: {source['metadata']}
{'-' * 50}
"""
    
    return formatted_response, response_text, sources

def evaluate_response(question: str, expected_response: str):
    """
    Evaluate if the RAG system's response matches the expected response
    
    Args:
        question (str): The question asked
        expected_response (str): The expected response
        
    Returns:
        bool: True if response matches expected, False otherwise
    """
    _, actual_response, _ = query_rag(question)
    
    # Simple evaluation - you can make this more sophisticated
    expected_lower = expected_response.lower()
    actual_lower = actual_response.lower()
    
    # Check if expected response is contained in actual response
    return expected_lower in actual_lower

# Example usage
if __name__ == "__main__":
    # Example query
    question = "What is the deployment process for the backend microservice?"
    formatted_response, response_text, sources = query_rag(question)
    print("\nQuery Results:")
    print(formatted_response)
    
    # Example evaluation
    expected = "docker build and kubernetes deployment"
    is_correct = evaluate_response(question, expected)
    print(f"\nEvaluation: Response matches expected? {is_correct}")