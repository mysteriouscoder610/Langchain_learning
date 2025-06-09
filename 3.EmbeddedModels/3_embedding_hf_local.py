from langchain_huggingface import HuggingFaceEmbedding

embedding = HuggingFaceEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')

text = "Delhi is the capital of India"

vector = embedding.embed_query(text)

print(str(vector))