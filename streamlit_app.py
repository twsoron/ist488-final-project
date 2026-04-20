import streamlit as st
from openai import OpenAI
from pathlib import Path
import chromadb

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

# Initialize ChromaDB client
if "collection" not in st.session_state:
    chroma_client = chromadb.PersistentClient(path="./ChromaDB")
    st.session_state.collection = chroma_client.get_or_create_collection("CourseCollection")

collection = st.session_state.collection

# Function that converts text into vector embeddings for semantic search
def embed(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Function to retrieve RAG context based on user question category
def retrieve_context(query, intent):
    query_embedding = embed(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where=FILTERS[intent]
    )
    if not results["documents"] or len(results["documents"][0]) == 0:
        return "No relevant course material found."
    
    docs = results["documents"][0]
    sources = results["metadatas"][0]

    context = "\n\n".join(
        f"Source: {src['source']}\n{doc}"
        for doc, src in zip(docs, sources)
    )

    return context

# Intent classifier function that uses a small model to classify user question into prompt categories
def classify_intent(question):
    response = client.responses.create(
        model="gpt-4o-mini",
        input=f"""
Classify the user's question into ONE of these categories:
- conceptual_question
- course_logistics
- debugging_help
- assignment_solution

Only return the label. No explanation.

Question: {question}
"""
    )

    intent = response.output[0].content[0].text.strip().lower() # Cleans output

    return intent

# Setup Streamlit App
st.title("IST 387 Assistant Chatbot")

# Stores chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

#Stores previous response for short-term memory
if "last_response_id" not in st.session_state:
    st.session_state.last_response_id = None

# Stores turn count for conversation context
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.last_response_id = None
    st.rerun()

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

    st.write("INTENT:", intent)
    st.write("CONTEXT:", context)

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