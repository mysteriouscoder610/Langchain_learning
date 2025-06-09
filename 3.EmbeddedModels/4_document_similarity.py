from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# You don't need dotenv unless you're loading something specific like Hugging Face API keys

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Virat Kohli is an Indian Cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian Captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "tell me about rohit sharma"

# Embed documents and query
doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

# Compute cosine similarities
scores=cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("Similarity score is:", score)

# Get most similar document
# most_similar_idx = np.argmax(similarities)
# print(f"Most similar document:\n{documents[most_similar_idx]}")
# print(f"Similarity Score: {similarities[most_similar_idx]}")
