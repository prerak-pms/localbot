from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,ServiceContext,PromptTemplate
from llama_index.llms.ollama import Ollama

from llama_index.embeddings.ollama.base import OllamaEmbedding

# Test embeddings
embed_model = OllamaEmbedding(model_name="mistral:latest")  # Using Ollama's local embedding
query_text = "March 10"
embedded_query = embed_model.get_text_embedding(query_text)
print("Query Embedding:", embedded_query)