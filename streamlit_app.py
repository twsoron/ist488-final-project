import json
import time

import streamlit as st
from openai import OpenAI
from pathlib import Path
import chromadb
import base64
from hybrid_search import HybridIndex
from reranker import rerank_safe
from r_executor import R_EXECUTOR_TOOL, handle_tool_call

def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()
img = get_base64("images/background.png")


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

# --- Long-term student memory ---------------------------------------------
# Sectioned per-student profile persisted to disk. Always-on slice is small
# (~200-400 tokens once populated) so context stays bounded. After each turn
# a cheap extractor proposes JSON patches and we merge with caps + dedupe.

MEMORY_DIR = Path(__file__).parent / "memory"
MEMORY_DIR.mkdir(exist_ok=True)

LIST_CAPS = {
    "learning_style": 5,
    "weak_areas": 5,
    "goals": 5,
    "recent_topics": 10,
}

EMPTY_PROFILE = {
    "profile": {},
    "learning_style": [],
    "progress": {},
    "weak_areas": [],
    "goals": [],
    "recent_topics": [],
    "last_updated": None,
}

EXTRACTOR_SYSTEM = """You maintain a student's long-term memory profile for a course chatbot.
Given the current profile and the latest exchange, return ONLY a JSON object with fields to add/update.

Allowed keys:
- profile: object of stable facts (name, major, year). Merged shallowly.
- learning_style: list of short phrases to append (e.g. "prefers worked examples")
- progress: object describing current course progress; values overwrite
- weak_areas: list of topics the student struggled with — append
- goals: list of stated goals — append
- recent_topics: list of topics discussed this turn — append

Rules:
- Return {} if nothing durable was learned.
- Each list item must be under 12 words.
- Only record things the student said or clearly demonstrated. No speculation.
- Do not duplicate items already in the profile."""


def memory_path(student_id):
    safe = "".join(c for c in student_id if c.isalnum() or c in "-_").lower()
    return MEMORY_DIR / f"{safe or 'anon'}.json"


def load_memory(student_id):
    p = memory_path(student_id)
    if p.exists():
        try:
            return {**EMPTY_PROFILE, **json.loads(p.read_text(encoding="utf-8"))}
        except json.JSONDecodeError:
            pass
    return {k: (v.copy() if isinstance(v, (list, dict)) else v) for k, v in EMPTY_PROFILE.items()}


def save_memory(student_id, mem):
    p = memory_path(student_id)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(mem, indent=2), encoding="utf-8")
    tmp.replace(p)


def render_memory_for_prompt(mem):
    visible = {k: v for k, v in mem.items() if k != "last_updated" and v}
    if not visible:
        return ""
    return (
        "\n\nSTUDENT MEMORY (use to personalize tone, examples, and depth — do not recite verbatim):\n"
        + json.dumps(visible, indent=2)
    )


def extract_updates(mem, user_msg, assistant_msg):
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": EXTRACTOR_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Current profile:\n{json.dumps(mem)}\n\n"
                        f"Student: {user_msg}\n\nAssistant: {assistant_msg}"
                    ),
                },
            ],
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        return {}


def merge_updates(mem, updates):
    new = {k: (v.copy() if isinstance(v, (list, dict)) else v) for k, v in mem.items()}
    for key, val in updates.items():
        if key in ("profile", "progress") and isinstance(val, dict):
            new[key] = {**new.get(key, {}), **val}
        elif key in LIST_CAPS and isinstance(val, list):
            existing = list(new.get(key, []))
            for item in val:
                if isinstance(item, str) and item.strip() and item not in existing:
                    existing.append(item.strip())
            new[key] = existing[-LIST_CAPS[key]:]
    new["last_updated"] = time.strftime("%Y-%m-%d")
    return new

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
st.text("Ask a question about homework, labs, functions/packages, or course information")

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

    st.markdown("### 👤 Student")
    student_id = st.text_input(
        "Student ID (optional — enables personalization)",
        key="student_id",
        placeholder="e.g. jjkomosi",
    )

    if student_id:
        if "memory" not in st.session_state or st.session_state.get("loaded_for") != student_id:
            st.session_state.memory = load_memory(student_id)
            st.session_state.loaded_for = student_id

        with st.expander("View my profile"):
            st.json(st.session_state.memory)

        if st.button("Clear my memory"):
            st.session_state.memory = {
                k: (v.copy() if isinstance(v, (list, dict)) else v)
                for k, v in EMPTY_PROFILE.items()
            }
            save_memory(student_id, st.session_state.memory)
            st.rerun()

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
    # Follow-ups like "what is that about?" don't carry enough signal for
    # BM25/vector retrieval on their own. When there's a prior user turn,
    # prepend it so routing + retrieval can resolve references.
    prior_user_msg = next(
        (m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"),
        None,
    )
    retrieval_query = f"{prior_user_msg}\n{question}" if prior_user_msg else question

    # Classify question intent using intent classifier function
    intent = classify_intent(retrieval_query)
    if intent not in PROMPTS:
        intent = "conceptual_question" # Even though model is programmed to choose out of a list of categories, ask it to revert to a conceptual question if it prints an unexpected label

    # Selects prompt instructions based on intent
    instructions = PROMPTS[intent]

    # Append student memory slice when a Student ID is set
    if student_id and "memory" in st.session_state:
        instructions += render_memory_for_prompt(st.session_state.memory)

    # Retrieve relevant course content using RAG
    context = retrieve_context(retrieval_query, intent)

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

        # Initial request — model may call run_r_code; we feed results back and
        # iterate up to MAX_TOOL_ROUNDS times before giving up.
        kwargs = dict(
            model="gpt-4o",
            instructions=instructions,
            input=f"""
                Use the following course material to help answer the question.

                Context:
                {context}

                Question:
                {question}
                """,
            tools=[R_EXECUTOR_TOOL],
            stream=True,
            store=True,
        )
        if st.session_state.last_response_id is not None:
            kwargs["previous_response_id"] = st.session_state.last_response_id

        MAX_TOOL_ROUNDS = 3
        for _ in range(MAX_TOOL_ROUNDS):
            stream = client.responses.create(**kwargs)
            function_calls = []

            for event in stream:
                if event.type == "response.output_text.delta":
                    full_text += event.delta
                    placeholder.write(full_text)
                elif event.type == "response.output_item.done":
                    item = event.item
                    if getattr(item, "type", None) == "function_call":
                        function_calls.append(item)
                elif event.type == "response.completed":
                    final_response = event.response

            if not function_calls:
                break

            # Execute tools and feed outputs back as the next input
            tool_outputs = []
            for fc in function_calls:
                try:
                    args = json.loads(fc.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_outputs.append({
                    "type": "function_call_output",
                    "call_id": fc.call_id,
                    "output": handle_tool_call(fc.name, args),
                })

            kwargs = dict(
                model="gpt-4o",
                previous_response_id=final_response.id,
                input=tool_outputs,
                tools=[R_EXECUTOR_TOOL],
                stream=True,
                store=True,
            )

        st.session_state.messages.append({"role": "assistant", "content": full_text}) # Store assistant response

        # Updates last response ID
        if final_response is not None:
            st.session_state.last_response_id = final_response.id

    # Update long-term memory after the response (non-blocking from user POV)
    if student_id and "memory" in st.session_state:
        updates = extract_updates(st.session_state.memory, question, full_text)
        if updates:
            st.session_state.memory = merge_updates(st.session_state.memory, updates)
            save_memory(student_id, st.session_state.memory)
