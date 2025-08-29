from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.llms.ollama import OllamaLLM

# 1. Ollama LLM (DeepSeek)
llm = OllamaLLM(model_name="deepseek-r1:1.5b")  # your Ollama model name

# 2. Ollama Embedding (or any Ollama embedding you have)
embed_model = OllamaEmbedding(model_name="nomic-embed-text")  # example embedding

# 3. Chroma setup
db = chromadb.PersistentClient(path="./my_chroma_db")
collection = db.get_or_create_collection("my_collection")
vector_store = ChromaVectorStore(chroma_collection=collection)

# 4. Load documents
documents = SimpleDirectoryReader("data").load_data()

# 5. Build index from embeddings
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
    vector_store=vector_store
)

# 6. Build query engine with DeepSeek LLM
query_engine = index.as_query_engine(llm=llm)

# 7. Query
response = query_engine.query("What is this about?")
print(response)
