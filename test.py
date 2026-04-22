import langchain

print(langchain.__version__)
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load API key
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize model
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama3-70b-8192"
)

# Chat loop
while True:
    user = input("You: ")

    if user.lower() == "exit":
        print("Bot: Goodbye 👋")
        break

    response = llm.invoke(user)
print("Bot:", response.content)