import streamlit as st
import uuid
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver

def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)

@st.cache_resource
def load_graph():
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        embedding_function=embeddings_model,
        persist_directory="./chroma_db",
        collection_name="dsa_book",
    )
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

    def rag_node(state: MessagesState):
        messages = state["messages"]
        latest_query = messages[-1].content
        prior_messages = messages[:-1]

        if prior_messages:
            history_text = "\n".join(
                f"{'User' if isinstance(m, HumanMessage) else 'Bot'}: {m.content}"
                for m in prior_messages
            )
            rewrite_prompt = f"""Given this conversation history and a follow-up question, rewrite the
follow-up into a standalone question. If it's already standalone, return it unchanged.
Return ONLY the rewritten question, nothing else.

History:
{history_text}

Follow-up question: {latest_query}

Standalone question:"""
            rewritten = extract_text(llm.invoke(rewrite_prompt).content).strip()
        else:
            rewritten = latest_query

        results = vectorstore.similarity_search(rewritten, k=3)
        context = "\n\n---\n\n".join(doc.page_content for doc in results)

        answer_prompt = f"""Use the following context from a Data Structures and Algorithms textbook to answer the question.
If the answer isn't in the context, say so honestly rather than guessing.

Context:
{context}

Question: {rewritten}

Answer:"""
        response = llm.invoke(answer_prompt)
        answer = extract_text(response.content)

        pages = sorted(set(doc.metadata['page'] for doc in results))
        answer_with_sources = f"{answer}\n\n*Sources: pages {pages}*"

        return {"messages": [AIMessage(content=answer_with_sources)]}

    builder = StateGraph(MessagesState)
    builder.add_node("rag", rag_node)
    builder.add_edge(START, "rag")
    builder.add_edge("rag", END)
    return builder.compile(checkpointer=InMemorySaver())

# --- Streamlit UI ---
st.title("DSA Book Chatbot")

graph = load_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a DSA question...")

if user_input:
    st.session_state.display_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
            answer = result["messages"][-1].content
        st.markdown(answer)

    st.session_state.display_messages.append({"role": "assistant", "content": answer})