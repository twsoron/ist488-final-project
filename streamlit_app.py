import streamlit as st
from openai import OpenAI

client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)

st.title("Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_response_id" not in st.session_state:
    st.session_state.last_response_id = None
    
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.last_response_id = None
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask a question")

if question:
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

        if st.session_state.last_response_id is None:
            stream = client.responses.create(
                model="gpt-4o",
                instructions="You are a helpful teaching assistant",
                input=question,
                stream=True,
                store=True
            )
        else:
            stream = client.responses.create(
                model="gpt-4o",
                instructions="You are a helpful teaching assistant",
                input=question,
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

        st.session_state.messages.append({"role": "assistant", "content": full_text})

        if final_response is not None:
            st.session_state.last_response_id = final_response.id