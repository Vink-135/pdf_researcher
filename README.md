# 🤖 AI PDF Chatbot (TensorFlow + Streamlit)

An intelligent chatbot that lets you **upload any PDF document** and ask **questions in natural language**.  
It finds the most relevant section of the document and uses **TensorFlow-powered T5** to generate accurate answers — just like you're chatting with your PDF.

---

## 📌 Features

- 📤 Upload any PDF
- ✂️ Chunk + Embed using `SentenceTransformers`
- 🔍 Find most relevant section with `cosine similarity`
- 🧠 Answer your query using `T5-small` (TensorFlow)
- 🖥️ Clean, interactive UI using **Streamlit**

---



## 🧰 Tech Stack

| Layer         | Tools Used                          |
|---------------|-------------------------------------|
| UI            | Streamlit                          |
| PDF Reading   | PyMuPDF (`fitz`)                    |
| Embeddings    | Sentence-Transformers (`MiniLM`)    |
| Similarity    | Cosine Similarity (`scikit-learn`)  |
| Q&A Model     | T5-small (`Transformers`, TensorFlow) |
| Language      | Python                              |

---

## 🛠 How It Works

1. **Upload PDF** → Extract full text using PyMuPDF
2. **Chunk the text** → Break it into 500-token pieces
3. **Generate embeddings** using `MiniLM`
4. **Compare query embedding** with all chunks
5. **Select the most similar chunk**
6. **Generate final answer** with T5 (`question: ... context: ...`)
7. **Streamlit displays it all beautifully!**

---

## 📂 Project Structure

```bash
pdf_chatbot/
├── app.py              # Streamlit app
├── pdf_reader.py       # Extract text from PDF
├── chunker.py          # Chunking and embedding logic
├── qa_answer.py        # TensorFlow T5 model logic
├── requirements.txt    # All dependencies
