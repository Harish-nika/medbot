import os
import numpy as np
import faiss
import pickle
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Directory for embeddings and text chunks
EMBEDDING_DIR = "/home/harish/Agentic_AI/embedding_chunk_bytitle"
TEXT_CHUNKS_DIR = "/home/harish/Agentic_AI/chunked_be_title"

# Load environment variables (API keys, etc.)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize ChatGroq LLM
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Llama3-8b-8192", temperature=0)

def load_embeddings():
    """
    Loads stored embeddings (.npy) and text chunks from their respective directories.
    """
    embeddings_list = []
    text_chunks = []

    files_found = [f for f in os.listdir(EMBEDDING_DIR) if f.endswith(".npy")]
    if not files_found:
        raise FileNotFoundError("⚠️ No `.npy` embedding files found!")

    for file in files_found:
        file_path = os.path.join(EMBEDDING_DIR, file)
        text_file = file.replace(".npy", "_chunks.pkl")
        text_path = os.path.join(TEXT_CHUNKS_DIR, text_file)

        try:
            embed = np.load(file_path)
            embeddings_list.append(embed)

            with open(text_path, "rb") as f:
                texts = pickle.load(f)
                text_chunks.extend(texts)
        except Exception as e:
            print(f"❌ Error loading {file}: {e}")
            continue

    if not embeddings_list:
        raise ValueError("❌ No valid embeddings found.")

    embeddings = np.vstack(embeddings_list)  # Stack embeddings
    return embeddings, text_chunks

def create_faiss_index(embeddings):
    """
    Creates a FAISS index for fast similarity search.
    """
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance metric for FAISS
    index.add(embeddings)  # Add embeddings to FAISS
    return index

# Load embeddings and create FAISS index
try:
    embeddings, text_chunks = load_embeddings()
    faiss_index = create_faiss_index(embeddings)
    print("✅ FAISS index created with", embeddings.shape[0], "entries.")
except Exception as e:
    print(f"❌ Error: {e}")

from sentence_transformers import SentenceTransformer

# Initialize your embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Use the model that fits your use case

def search_faiss_index(query, faiss_index, embeddings, text_chunks, k=5):
    """
    Perform a search on the FAISS index for the most similar embeddings to the query.
    """
    # Convert query to embedding using the same model
    query_embedding = embedding_model.encode([query])  # Make sure embedding_model is defined

    # Perform the FAISS search
    distances, indices = faiss_index.search(np.array(query_embedding).astype(np.float32), k)
    
    # Retrieve the most similar text chunks
    top_k_chunks = [text_chunks[i] for i in indices[0]]
    
    return top_k_chunks, distances[0]


def create_chatgroq_prompt(top_k_chunks, query):
    """
    Creates a formatted prompt for ChatGroq using retrieved chunks.
    """
    context = "\n".join(top_k_chunks)
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the following question based on the provided context:
        <context>
        {context}
        </context>
        Question: {input}
        """
    )
    
    formatted_prompt = prompt.format(context=context, input=query)
    return formatted_prompt

def query_chatgroq_with_context(query, faiss_index, embeddings, text_chunks, top_k=5):
    """
    Query ChatGroq with the enhanced context retrieved from FAISS search.
    """
    # Step 1: Retrieve the relevant chunks from FAISS
    top_k_chunks, distances = search_faiss_index(query, faiss_index, embeddings, text_chunks, k=top_k)

    # Step 2: Prepare the prompt for ChatGroq
    formatted_prompt = create_chatgroq_prompt(top_k_chunks, query)

    # Step 3: Query ChatGroq with the enhanced prompt
    response = llm.invoke(formatted_prompt)  # Use the correct method to invoke

    # Debugging: Print the response and its type
    print(f"Response from ChatGroq: {response}")
    print(f"Response type: {type(response)}")  # Check the type of response

    # Assuming the answer is stored in the 'content' attribute
    return response.content  # Or adjust if a different attribute is used

# Example query
query = "what are the heart diseases that can be treated with surgery?"

# Query ChatGroq with relevant context from FAISS
answer = query_chatgroq_with_context(query, faiss_index, embeddings, text_chunks, top_k=5)

print(f"Answer from ChatGroq: {answer}")

