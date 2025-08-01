from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from openai import OpenAI
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_openai import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
import json
from langchain.docstore.document import Document
from langchain.prompts.chat import ChatPromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import re
from langchain_core.messages import HumanMessage
from langchain.chat_models import ChatOpenAI
import os 
from dotenv import load_dotenv
load_dotenv(".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -------------------------------------
# Step 1: Q&A Data from JSON
# -------------------------------------
with open("merged_data.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# Combine each Q&A pair into a single document
documents = []
for qa in qa_data:
    content = f"Question: {qa[0]}\nAnswer: {qa[1]}"
    documents.append(Document(page_content=content))

# 1. Configure a splitter that only subdivides if a chunk exceeds 1000 chars.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # target <=1 000 chars per chunk
    chunk_overlap=200,      # keep 200 chars overlap for context
    length_function=len,    # measure by character count
)

# 2. Apply it—short docs (under 1000 chars) remain intact; long ones get split.
chunked_docs = text_splitter.split_documents(documents)

# Use a free embedding model from Hugging Face (e.g., "all-MiniLM-L6-v2")
#embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Create the vector store using FAISS
#vectorstore = FAISS.from_documents(documents, embedding_model)
vectorstore = Chroma.from_documents(documents=documents, embedding=embedding_model)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.4,
    openai_api_key=OPENAI_API_KEY,
)


def print_answer(answer):
    # Split on end‑of‑sentence punctuation (keeping the punctuation)
    sentences = re.split(r'(?<=[\?\.\!])\s+', answer.strip())

    # Print each sentence on its own line
    for s in sentences:
        print(s)
def gen_llm():
    contextualize_q_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\{context}"""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = """
        You are a senior technical interviewer.

        Ask from user an interview question, based on the Relevant Knowledge‑Base Excerpts and given input:
        {input}
        if you don't find a question related to the input, just say that you don't know.
        {context}
        Question:
        """
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain


def evaluate_llm():
    contextualize_q_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.{context}'"""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = """
        You are a senior technical interviewer assistant.

        Interview Question:
        {input}

        Candidate’s Answer:
        {user_answer}

        Relevant Knowledge‑Base Excerpts:
        {context}
        Evaluate the candidate’s answer based on the following criteria:
            a) Score the candidate’s answer from 1 (poor) to 10 (excellent).  
            b) Provide concise, actionable feedback on how to improve.  
            c) Supply the “model” (ideal) answer.  
        Format your response as:
            Score: <number>  
            Feedback: <text>  
            Model Answer: <text>

        Begin now.
        """
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("system", "{input}"),
            ("human", "Candidate’s Answer: {user_answer}")
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

def communicate_llm():
    contextualize_q_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\{context}"""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = """
        You are a senior technical interviewer.
        communicate with the user, based on the Relevant Knowledge‑Base Excerpts and given input and return Answer to the users:
        {input}
        if you don't find a question related to the input, just say that you don't know.
        {context}
        Answer:
        """
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

'''''
chat_history = []
first_question = "Generate new question about data science"
generated_question = gen_llm().invoke({"input":first_question,"chat_history": chat_history})
chat_history.extend([HumanMessage(content=first_question), generated_question["answer"]])
response = generated_question['answer']
print_answer(response)

answer = """Ptorch, scikit learn, numpy and pandas Library."""
evaluate = evaluate_llm().invoke({"input": response, "context": "", "user_answer": answer, "chat_history": chat_history})
chat_history.extend([HumanMessage(content=answer), evaluate["answer"]])
print_answer(evaluate["answer"])
'''''