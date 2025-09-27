# qa_app.py

import streamlit as st
import torch
from transformers import pipeline
from datasets import load_dataset
import random

# --- streamlit page configuration ---
st.set_page_config(page_title="QA Chatbot", layout="wide")
st.title("🤖 Question Answering Chatbot")
st.markdown("Ask any question based on a context — either from examples or your own text.")


# --- load qa model and dataset ---
@st.cache_resource
def load_qa_pipeline():
    # Load model
    model_name = "distilbert-base-uncased-distilled-squad"
    qa_pipe = pipeline("question-answering", model=model_name, framework="pt")
    return qa_pipe

# --- load sample contexts from SQuAD dataset ---
@st.cache_data
def load_sample_contexts():
  # Load a few sample contexts from the SQuAD dataset
  dataset = load_dataset("squad")
  train_split = dataset["train"]
  random_indices = random.sample(range(len(train_split)), 10)
  selected_examples = train_split.select(random_indices)

  # Extract contexts, questions, and answers
  contexts = []
  for item in selected_examples:
    contexts.append({
      "context": item["context"],
      "question": item["question"],
      "answers": item["answers"]
    })
  return contexts

# --- main app logic ---
def main():
  qa_pipeline = load_qa_pipeline()

  # modes
  tab1, tab2 = st.tabs(["📚 Example Mode", "✍️ Custom Context Mode"])

  ## ------ Example Mode ------ ##
  with tab1:
    st.subheader("📘 Ask questions based on sample SQuAD contexts")

    # Example context selection
    sample_data = load_sample_contexts()
    context_options = [f"Example {i+1}" for i in range(len(sample_data))]
    selected_idx = st.selectbox(
      "Choose a sample context",
      options=range(len(context_options)),
      format_func=lambda x: context_options[x]
    )

    # Display selected context
    selected_data = sample_data[selected_idx]
    context = selected_data["context"]
    with st.expander("📜 Current Context"):
      st.write(context)

    # Ask question
    user_question = st.text_input("🔍 Ask a question based on the above context:", key="example_q")

    # Process QA
    if user_question:
      try:
        result = qa_pipeline(question=user_question, context=context)
        answer = result['answer']
        st.markdown(
          f""" <div style="background-color:#003300;padding:10px;border-radius:8px">
            <span style="color:#00FF99;font-weight:bold;">✅ Answer:</span><br>
            <span style="color:white;font-size:16px;">{answer}</span>
          </div>""",
          unsafe_allow_html=True )
      except Exception as e:
        st.error("❌ Error while processing the question.")
        st.exception(e)

    # Reference question
    with st.expander("💡 Reference Example (from SQuAD)"):
      st.markdown(f"**Question:** {selected_data['question']}")
      st.markdown(f"**Expected Answer(s):** {', '.join(selected_data['answers']['text'])}")

  
  ## ------ Custom Context Mode ------ ##
  with tab2:
    st.subheader("📝 Paste your own context and ask questions")

    # User inputs custom context and question
    custom_context = st.text_area("📄 Paste your context here:", height=200)
    custom_question = st.text_input("🔍 Ask a question based on your custom context:", key="custom_q")

    # Process QA
    if custom_context and custom_question:
      try:
        result = qa_pipeline(question=custom_question, context=custom_context)
        answer = result['answer']
        st.markdown(
          f""" <div style="background-color:#002B36;padding:10px;border-radius:8px">
            <span style="color:#00FF99;font-weight:bold;">✅ Answer:</span><br>
            <span style="color:white;font-size:16px;">{answer}</span> </div> """,
          unsafe_allow_html=True)
      except Exception as e:
        st.error("❌ Error while processing your question.")
        st.exception(e)
    elif custom_context and not custom_question:
      st.info("💡 Please enter a question.")
    elif custom_question and not custom_context:
      st.warning("⚠️ Please paste a context first.")

# Run Streamlit App
if __name__ == "__main__":
  main()
