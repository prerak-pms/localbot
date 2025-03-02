from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,ServiceContext,PromptTemplate
from llama_index.llms.ollama import Ollama

from llama_index.embeddings.ollama.base import OllamaEmbedding

# Load documents from the current directory
# Check if documents are loaded correctly
documents = SimpleDirectoryReader(input_dir=".").load_data()
print(f"Loaded {len(documents)} documents.")
for doc in documents:
    print(doc.text[:500])  # Print first 500 characters of each doc

# Use Ollama for both LLM and embeddings
llm = Ollama(model="mistral:latest", request_timeout=180)
embed_model = OllamaEmbedding(model_name="mistral:latest")  # Using Ollama's local embedding

# Create an index from documents
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Query the document
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("When is the website launch?")

print(response)