
import os

from dotenv import load_dotenv

# Enable NeMo's LangChain integration
load_dotenv()
os.environ["NEMOGUARDRAILS_LLM_FRAMEWORK"] = "langchain"

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails


# Step 1: Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Step 2: Create the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Answer clearly and briefly."),
    ("human", "{question}")
])

# Step 3: Load NeMo configuration
config = RailsConfig.from_path("config")

# Step 4: Initialize guardrails
guardrails = RunnableRails(config, passthrough=True)

# Step 5: Build LangChain with guardrails
chain = (
    prompt
    | (guardrails | llm)
    | StrOutputParser()
)


# Step 6: Ask a question
def ask_ai(question):
    return chain.invoke({"question": question})


# Step 7: Interactive application
if __name__ == "__main__":
    while True:
        question = input("\nEnter your question (exit to quit): ")

        if question.lower() == "exit":
            break

        answer = ask_ai(question)
        print("\nAI Response:", answer)
