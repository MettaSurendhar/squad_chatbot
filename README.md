# 🤖 SQuAD chatbot

An interactive **Question Answering Chatbot** built with **Streamlit**, **Hugging Face Transformers**, and the **SQuAD** dataset.  
You can either:

- 🧪 Choose a random sample context from SQuAD and ask questions
- ✍️ Paste your own custom text and ask any question

---

## 🧰 Tech Stack

- [Streamlit](https://streamlit.io/) — for building the web app
- [Hugging Face Transformers](https://huggingface.co/transformers/) — for the QA model
- [Datasets](https://huggingface.co/docs/datasets/) — to load SQuAD
- [DistilBERT](https://huggingface.co/distilbert-base-uncased-distilled-squad) — pre-trained model for question answering
- Python 3.9+

---

## 🚀 Features

✅ Choose from 10 random SQuAD examples  
✅ Ask questions based on selected context  
✅ Paste your own context and ask any question  
✅ Clean UI with styled answer display  
✅ Cached model & data for faster performance

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/streamlit-qa-chatbot.git
cd streamlit-qa-chatbot
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### ▶️ Run the App

```bash
streamlit run qa_app.py
```

### 📁 Project Structure

```bash
streamlit-qa-chatbot/
│
├── qa_app.py            # Main Streamlit app
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

### 📜 requirements.txt (example)

```bash
streamlit
transformers
torch
datasets
```

### 🧠 Example Usage

1. Example Mode

- Choose from 10 random SQuAD contexts
- Ask a question and view the model’s answer

2. Custom Context Mode

- Paste your own text
- Ask any question to get an answer
