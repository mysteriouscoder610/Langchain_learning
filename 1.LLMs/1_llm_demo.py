from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Initialize the Gemini LLM - Choose from available free models
llm = GoogleGenerativeAI(
    model="gemini-2.0-flash"     # Latest and fastest (recommended)
    # model="gemini-2.0-flash-lite", # Cost-optimized, faster, lighter version
    # model="gemini-1.5-flash",      # Stable alternative (being phased out)
    # model="gemini-1.5-pro",        # More capable but slower (being phased out)
    # model="gemini-2.5-flash",      # Newest with thinking capabilities
    # temperature=0.7,
    # max_output_tokens=1024,
    # top_p=0.8,
    # top_k=40
)

# Invoke the LLM with your question
result = llm.invoke("What is the capital of India?")

print(result)