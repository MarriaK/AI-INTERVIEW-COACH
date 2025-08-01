import streamlit as st
from langchain_core.messages import HumanMessage
from RagMain import gen_llm, evaluate_llm, communicate_llm

st.set_page_config(page_title="RAG Interviewer System", layout="wide")
st.title("💼👔 RAG Interviewer System")

# Initialize histories & flags
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "interview_history" not in st.session_state:
    st.session_state.interview_history = []
if "interview_awaiting_answer" not in st.session_state:
    st.session_state.interview_awaiting_answer = False

tab_chat, tab_interview = st.tabs(["💬 Chat", "💼 Interview"])

# Helper to render a history container
def render_chat(container, history):
    for msg in history:
        role = "assistant" if msg["role"] == "assistant" else "user"
        with container:
            st.chat_message(role).write(msg["content"])

# ——— Tab 1: General Chat ———
with tab_chat:
    st.header("General Chat")

    chat_container = st.container()
    render_chat(chat_container, st.session_state.chat_history)

    # Input at bottom
    user_text = st.chat_input("Ask me anything...")
    if user_text:
        # User message
        st.session_state.chat_history.append({"role": "user", "content": user_text})
        with chat_container:
            st.chat_message("user").write(user_text)

        # Assistant reply
        comm_chain = communicate_llm()
        comm_out = comm_chain.invoke({
            "input": user_text,
            "chat_history": [HumanMessage(content=m["content"])
                              for m in st.session_state.chat_history],
            "context": ""
        })
        reply = comm_out["answer"].strip()
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        with chat_container:
            st.chat_message("assistant").write(reply)

# ——— Tab 2: Interview ———
with tab_interview:
    st.header("Interview Q&A")

    interview_container = st.container()
    render_chat(interview_container, st.session_state.interview_history)

    prompt = "Type a topic to generate a question, or answer the last one."
    user_text = st.chat_input(prompt, key="interview_input")
    if user_text:
        # User turn
        st.session_state.interview_history.append({"role": "user", "content": user_text})
        with interview_container:
            st.chat_message("user").write(user_text)

        if not st.session_state.interview_awaiting_answer:
            # Generate question
            gen_chain = gen_llm()
            gen_out = gen_chain.invoke({
                "input": user_text,
                "chat_history": [HumanMessage(content=m["content"])
                                  for m in st.session_state.interview_history],
                "context": ""
            })
            question = gen_out["answer"].strip()
            st.session_state.interview_history.append({"role": "assistant", "content": question})
            with interview_container:
                st.chat_message("assistant").write(question)
            st.session_state.interview_awaiting_answer = True

        else:
            # Evaluate answer
            last_q = next(
                (m["content"] for m in reversed(st.session_state.interview_history)
                 if m["role"] == "assistant"),
                ""
            )
            eval_chain = evaluate_llm()
            eval_out = eval_chain.invoke({
                "input": last_q,
                "user_answer": user_text,
                "chat_history": [HumanMessage(content=m["content"])
                                  for m in st.session_state.interview_history],
                "context": ""
            })
            feedback = eval_out["answer"].strip()
            st.session_state.interview_history.append({"role": "assistant", "content": feedback})
            with interview_container:
                st.chat_message("assistant").write(feedback)
            st.session_state.interview_awaiting_answer = False