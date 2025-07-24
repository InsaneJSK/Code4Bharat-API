import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def chat_title(user_input, llm_response):
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    CHAT_PROMPT = ChatPromptTemplate.from_template("""
    You are a helpful assistant that has to give a suitable title to the chat provided between a user and an llm.
    Make sure your title is only 4-5 words.
    User:
    {user_input}
    llm:
    {llm_response}
    """)
    chain = CHAT_PROMPT|llm
    response = chain.invoke(
        {
            "user_input" : user_input,
            "llm_response" : llm_response
        }
    )
    output = response.content 
    return output

if __name__ == "__main__":
    user_input = input("Enter user input: ")
    llm_response = input("Enter llm response: ")
    print(chat_title(user_input=user_input, llm_response=llm_response))