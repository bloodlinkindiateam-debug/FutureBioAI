
import json
import gradio as gr
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load knowledge
with open("futurebio_knowledge.json", "r", encoding="utf-8") as f:
    unified_db = json.load(f)

# Load AI
ai = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

# Build search index
documents = [item["content"] for item in unified_db]

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

document_vectors = vectorizer.fit_transform(documents)


def smart_search(question, top_k=3):
    query_vector = vectorizer.transform([question])

    similarities = cosine_similarity(
        query_vector,
        document_vectors
    )[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []

    for index in top_indices:
        results.append(unified_db[index])

    return results


def ask_ai(question, source, history):

    if not question.strip():
        return history, ""

    results = smart_search(question, 5)

    if source != "All":
        results = [
            r for r in results
            if r["source"] == source
        ]

    if not results:
        answer = "Relevant information nahi mili."
    else:

        context = "\n\n".join([
            f"Source: {r['source']}\n"
            f"Section: {r['section']}\n"
            f"Information: {r['content']}"
            for r in results[:3]
        ])

        prompt = f"""
You are FutureBioAI, an educational Biology assistant.

Question:
{question}

Study material:
{context}

Give a simple educational answer.
Do not invent unsupported facts.
If information is insufficient, say so.
Keep biology explanations safe and high-level.

Answer:
"""

        result = ai(
            prompt,
            max_new_tokens=200
        )

        answer = result[0]["generated_text"]

    history = history or []

    history.append({
        "role": "user",
        "content": question
    })

    history.append({
        "role": "assistant",
        "content": answer
    })

    return history, ""


def clear_chat():
    return []


nios_count = sum(
    1 for x in unified_db
    if x["source"] == "NIOS Biology"
)

research_count = sum(
    1 for x in unified_db
    if x["source"] == "Research PDF"
)

total_count = len(unified_db)


with gr.Blocks(
    title="FutureBioAI"
) as app:

    gr.Markdown("""
    # 🧬 FutureBioAI
    ### Rajneesh Chaturvedi
    **Biology Research & Learning Assistant**
    """)

    with gr.Row():

        gr.Markdown(
            f"### 📚 NIOS Biology\n## {nios_count}"
        )

        gr.Markdown(
            f"### 🔬 Research\n## {research_count}"
        )

        gr.Markdown(
            f"### 🧠 Knowledge\n## {total_count}"
        )

    source = gr.Radio(
        ["All", "NIOS Biology", "Research PDF"],
        value="All",
        label="📖 Knowledge Source"
    )

    chatbot = gr.Chatbot(
        label="FutureBioAI",
        height=500
    )

    question = gr.Textbox(
        label="Ask your Biology question",
        placeholder="Type your question...",
        lines=3
    )

    with gr.Row():

        ask_button = gr.Button("🚀 Ask FutureBioAI")
        clear_button = gr.Button("🗑️ Clear Chat")

    ask_button.click(
        ask_ai,
        inputs=[question, source, chatbot],
        outputs=[chatbot, question]
    )

    question.submit(
        ask_ai,
        inputs=[question, source, chatbot],
        outputs=[chatbot, question]
    )

    clear_button.click(
        clear_chat,
        outputs=chatbot
    )

    gr.Markdown("""
    ---
    **FutureBioAI • Built by Rajneesh Chaturvedi**
    """)


app.launch()
