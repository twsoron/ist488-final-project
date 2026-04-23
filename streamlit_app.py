import streamlit as st
from openai import OpenAI
from pathlib import Path
import chromadb
import base64
from hybrid_search import HybridIndex
from reranker import rerank_safe

def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()
img = get_base64("images/background.png")

st.markdown(
    f"""
    <style>
    .stApp {{
        background:
            linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)),
            url("data:image/jpg;base64,{img}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    div[data-testid="stChatInput"] {
        background-color: transparent !important;
    }

    div[data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Reads prompt MD files with instructions for how to answer different types of student questions
def read_prompt(file_name):
    return (Path("prompts") / file_name).read_text()

# Directs the model to the right instructions prompt
PROMPTS = {
    "conceptual_question": read_prompt("conceptual_prompt.md"),
    "course_logistics": read_prompt("course_logistics.md"),
    "debugging_help": read_prompt("debugging_help.md"),
    "assignment_solution": read_prompt("solution_request.md"),
}

# RAG retrieval filters that ensure that the question is mapped to a metadata filter used in ChromaDB retrieval
FILTERS = {
    "conceptual_question": {"type": "concept"},
    "course_logistics": {"type": "syllabus"},
    "debugging_help": {"type": "concept"},
    "assignment_solution": {"type": "assignment"},
}

# Create client session state
if "client" not in st.session_state:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.session_state.client = OpenAI(api_key=api_key)

client = st.session_state.client

# Initialize ChromaDB client + in-memory BM25 index for hybrid search.
if "collection" not in st.session_state:
    chroma_client = chromadb.PersistentClient(path="./ChromaDB")
    st.session_state.collection = chroma_client.get_or_create_collection("CourseCollection")

collection = st.session_state.collection

if "hybrid_index" not in st.session_state:
    st.session_state.hybrid_index = HybridIndex(collection)

hybrid_index = st.session_state.hybrid_index

# Hybrid (BM25+vector) retrieval -> ZeroEntropy rerank. Drop candidates below
# this reranker score so weakly-relevant context doesn't poison the answer.
RERANK_SCORE_FLOOR = 0.25
FETCH_K = 15
RERANK_TOP_N = 5

def embed(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def retrieve_context(query, intent):
    query_embedding = embed(query)

    fused = hybrid_index.hybrid_retrieve(
        query=query,
        query_embedding=query_embedding,
        top_k=FETCH_K,
        fetch_k=FETCH_K,
        where=FILTERS.get(intent),
    )
    if not fused:
        return "No relevant course material found."

    ranked = rerank_safe(query, fused, top_n=RERANK_TOP_N)

    # rerank_safe returns all-zero scores when it falls back (e.g. missing
    # ZEROENTROPY_API_KEY). Only apply the relevance floor when the reranker
    # actually ran — otherwise we'd drop every chunk and answer nothing.
    reranker_ran = any(r.get("relevance_score", 0) > 0 for r in ranked)
    if reranker_ran:
        ranked = [r for r in ranked if r.get("relevance_score", 0) >= RERANK_SCORE_FLOOR]

    if not ranked:
        return "No relevant course material found."

    return "\n\n".join(
        f"Source: {r.get('source')}\n{r['text']}" for r in ranked
    )

# Intent classifier function that uses a small model to classify user question into prompt categories
def classify_intent(question):
    response = client.responses.create(
        model="gpt-5.4-mini",
        input=f"""Classify the question into ONE label:
- conceptual_question: asking what a concept, function, or package does
- course_logistics: asking about deadlines, schedule, grading, or syllabus policy
- debugging_help: asking why code errors or how to fix a bug
- assignment_solution: asking for help solving a specific homework or lab problem

Questions that mention "HW" or "homework" can go either way — route by WHAT is being asked, not just the keyword:
- "When is HW 3 due?" -> course_logistics
- "What is part B of HW 3 asking me to do?" -> assignment_solution
- "How do I solve problem 2 of HW 5?" -> assignment_solution

Return only the label, nothing else.

Question: {question}"""
    )

    intent = response.output[0].content[0].text.strip().lower()

    return intent

# Setup Streamlit App
st.title("IST 387 Assistant Chatbot")
st.caption("Ask a question about homework, labs, functions/packages, or course information")

with st.sidebar:
    st.markdown("### 📚 IST 387 Assistant")
    st.caption("A Socratic learning companion for Applied Data Science")
    st.divider()
    with st.expander("💡 How to use"):
        st.markdown("""
- Ask concept questions about R, stats, or data science
- Ask about course logistics like deadlines or grading
- Paste broken R code for debugging help
- Ask for guidance on assignments — this bot guides, not gives answers
        """)
    st.divider()
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.last_response_id = None
        st.rerun()
        
# Stores chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

#Stores previous response for short-term memory
if "last_response_id" not in st.session_state:
    st.session_state.last_response_id = None

# Stores turn count for conversation context
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
question = st.chat_input("Ask a question")

if question:
    # Classify question intent using intent classifier function
    intent = classify_intent(question)
    if intent not in PROMPTS:
        intent = "conceptual_question" # Even though model is programmed to choose out of a list of categories, ask it to revert to a conceptual question if it prints an unexpected label

    # Selects prompt instructions based on intent
    instructions = PROMPTS[intent]

    # Retrieve relevant course content using RAG
    context = retrieve_context(question, intent)

    # Manage conversation length
    st.session_state.turn_count += 1
    if st.session_state.turn_count > 10:
        st.session_state.last_response_id = None
        st.session_state.turn_count = 0

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""
        final_response = None

        # Calls model with no previous memory
        if st.session_state.last_response_id is None:
            stream = client.responses.create(
                model="gpt-4o",
                instructions=instructions,
                input=f"""
                    Use the following course material to help answer the question.

                    Context:
                    {context}

                    Question:
                    {question}
                    """,
                stream=True,
                store=True
            )
        else: # Continues a conversation that already has memory
            stream = client.responses.create(
                model="gpt-4o",
                instructions=instructions,
                input=f"""
                    Use the following course material to help answer the question.

                    Context:
                    {context}

                    Question:
                    {question}
                    """,
                previous_response_id=st.session_state.last_response_id,
                stream=True,
                store=True
            )

        for event in stream:
            if event.type == "response.output_text.delta":
                full_text += event.delta
                placeholder.write(full_text)
            elif event.type == "response.completed":
                final_response = event.response

        st.session_state.messages.append({"role": "assistant", "content": full_text}) # Store assistant response

        # Updates last response ID
        if final_response is not None:
            st.session_state.last_response_id = final_response.id
