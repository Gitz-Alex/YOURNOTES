import pyfiglet
from simple_chalk import chalk
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from build_knowledgebase import build_vectorstore, load_vectorstore

load_dotenv()


# --- Hybrid QA function ---
def ask_question(query, vectorstore, k=3):
    relevant_docs = vectorstore.similarity_search(query, k=k)
    context_text = "\n".join([doc.page_content for doc in relevant_docs])

    prompt_template = PromptTemplate(
        input_variables=["context", "query"],
        template=(
            "You are an AI assistant.\n"
            "Use the following context to answer the question. and add minimal examples if neccessary "
            "If the answer is not in the context, answer based on your own knowledge and refer the user to reliable sources.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{query}\n\n"
            "Answer only in a concise and informative manner."
        ),
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
    chain = RunnableSequence(prompt_template | llm | StrOutputParser())
    response = chain.invoke({"context": context_text, "query": query})
    return response


# --- Interactive CLI ---
if __name__ == "__main__":
    # load vectorestore
    vectorstore = load_vectorstore()
    print(chalk.green(pyfiglet.figlet_format("YOURNOTES")))
    print("Ask a question based on your knowledge base buddy. Type 'exit' to quit.\n")

    while True:
        query = input("Ask anything: ").strip()
        if query.lower() in ("exit", "quit"):
            print(chalk.green(pyfiglet.figlet_format("SEEYUH !")))
            break
        if not query:
            continue

        response = ask_question(query, vectorstore)
        print(f"Response: {response}\n")
