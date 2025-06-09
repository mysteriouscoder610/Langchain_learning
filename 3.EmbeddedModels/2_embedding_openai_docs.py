from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

documents = [
    "Delhi is the Capital of india",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

result = embedding.embed_documents("Delhi is the Capital of india")
# upar wale function hai multiple documents ka query generate krne ke lie
# the above line will give you 32 dimensional contextual vector of the above sentence

print(str(result))