# 🧑‍🍳 Bengali Masterchef – A RAG-based Cooking Assistant in Bengali

This is a **Retrieval-Augmented Generation (RAG)** application that acts as a **Bengali cooking masterchef**, helping users with authentic Bengali recipes in their native language.

Built using **GPT-4o**, **OpenAI Embedding Model**, **Pinecone**, **LangChain**, and **Streamlit**.

---

## 🚀 Features

- 🧠 Powered by **OpenAI GPT-4o**
- 🗣️ Responds in natural **Bengali language**
- 📚 Uses vector similarity search to fetch cooking-related context
- 🍛 Returns contextualized Bengali recipes
- 🖼️ Chat-style UI with persistent history
- ✅ Stateless deploy with Streamlit Cloud

---

## 🛠️ Tech Stack

| Component          | Tech Used                        |
|-------------------|----------------------------------|
| LLM               | OpenAI GPT-4o                    |
| Embedding Model   | `text-embedding-3-large`         |
| Vector DB         | Pinecone                         |
| Backend Framework | LangChain                        |
| UI                | Streamlit                        |
| Deployment        | Streamlit Community Cloud        |

---

## ⚙️ Technical Details

### 🔡 Embedding Model

- **Model:** `text-embedding-3-large`
- **Dimensions:** `3072`
- **Context window:** 8192 tokens

### 🔍 Retrieval System

- **Vector Store:** Pinecone
- **Search Type:** `similarity`
- **Top-K:** `5` most relevant chunks are retrieved per query