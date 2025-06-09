from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Initialize the Gemini chat model
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Latest and fastest model
    # model="gemini-1.5-flash",  # Alternative stable model
    # model="gemini-1.5-pro",   # More capable model
    temperature=0.9,
    max_output_tokens=10
    # convert_system_message_to_human=True
    # Temperature - It is a parameter that controls the randomness of a language 
                # model's output, it affects how creative or deterministic the responses are.
                # Lower Values: (0.0 - 0.3) -> More deterministic and predictable
                #  Higher Values: (0.7 - 1.5) -> More random, creative and diverse
)

# Invoke the model with your question
result = model.invoke("Write a 5 line poem on cricket")

print(result.content)