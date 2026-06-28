# 🧠 Bharat-Law-AI

### *Agentic AI for Intelligent Legal Document Retrieval & Multilingual Legal Query Understanding*

In Enlglish Language

<img width="1918" height="957" alt="image" src="https://github.com/user-attachments/assets/f14dc3fb-b921-48e2-927d-fdeeacbd11b7" />


In Tamil Language

<img width="1916" height="911" alt="image" src="https://github.com/user-attachments/assets/f3f58759-b70d-4dbf-8964-43c43c609c18" />

---

## 📘 Overview

**Bharat-Law-AI** is an advanced **Agentic AI system** designed for the **Indian legal ecosystem**, integrating **Retrieval-Augmented Generation (RAG)**, **multi-agent architectures**, and **multilingual intelligence**. The project aims to provide **context-aware legal information retrieval** and **smart query handling** across complex document formats and multiple Indian languages.

This system enables efficient **legal document search**, **summarization**, and **contextual reasoning** using dynamic vector store selection and optimized data retrieval strategies — all powered by **Gemini LLM** and **FAISS**.

---

## 🚀 Key Features & Novel Contributions

### 🧩 1. Agentic AI with Dynamic Vector Store Selection

* Developed an **Agentic AI framework** capable of dynamically selecting the **optimal vector store** for **fast** and **token-efficient** RAG-based legal retrieval.
* This adaptive mechanism ensures minimal latency and optimized token usage during context retrieval and reasoning.

### 💾 2. FAISS Vector Stores in SQL Database

* **FAISS vector stores** are persistently saved and indexed in an **SQL database**, enabling seamless **storage**, **retrieval**, and **scalable document management**.
* This hybrid integration accelerates search operations while maintaining a structured backend.

### 🧑‍⚖️ 3. Multi-Agent LLM Architecture (Gemini-based)

* Built a **multi-agent LLM architecture** using **Google’s Gemini** model to handle **contextual legal queries** efficiently.
* Agents collaborate for **document retrieval**, **context interpretation**, **summarization**, and **cross-lingual translation** of legal texts.

### 📄 4. End-to-End Data Extraction Pipeline

* Implemented a comprehensive **data extraction pipeline** using:

  * **PyMuPDF** and **pdfplumber** for parsing complex PDF layouts
  * **OpenCV** and **OCR** for structured and unstructured document extraction from scanned legal documents and images
* The pipeline supports **PDF, DOCX, and image** formats for comprehensive input coverage.

### 🌐 5. Multilingual Legal Testing Framework

* Designed a **robust multilingual testing framework** covering **50 diverse and complex legal scenarios** across **6 Indian languages**.
* Achieved an **86.75% similarity accuracy**, validating the system’s ability to interpret and respond accurately in multilingual environments.

---

## 🧮 Tech Stack

| Category                 | Technologies Used                             |
| ------------------------ | --------------------------------------------- |
| **LLMs & RAG**           | Gemini Large Language Model, LangChain, FAISS |
| **Database**             | SQL (for FAISS vector storage)                |
| **Document Parsing**     | PyMuPDF, pdfplumber, OpenCV, OCR              |
| **Backend**              | Python                                        |
| **Embeddings**           | LaBSE                                         |
| **Testing & Evaluation** | Multilingual similarity and accuracy metrics  |

---

## ⚙️ System Workflow

1. **Document Ingestion:** Upload legal documents (PDF, DOCX, Images).
2. **Parsing & Extraction:** Processed through PyMuPDF, pdfplumber, and OCR pipeline.
3. **Embedding Generation:** Generate embeddings using **LaBSE** and store them in SQL-integrated FAISS vector stores.
4. **Dynamic Retrieval:** Agentic AI selects the most relevant vector store based on query context.
5. **Query Processing:** Gemini-based multi-agent LLM processes and synthesizes context-aware responses.
6. **Multilingual Output:** Responses are tested across multiple Indian languages for robustness and semantic accuracy.

---

## 📊 Evaluation

* **Scenarios Tested:** 50
* **Languages Covered:** Hindi, Tamil, Bengali, Telugu, Kannada, Marathi
* **Similarity Accuracy:** 86.75%
* **Performance Metric:** Token efficiency, retrieval latency, and multilingual coherence

---

## 🧠 Future Enhancements

* Integration with **court database APIs** for real-time case retrieval.
* Adding **speech-to-text support** for verbal legal queries.
* Enhancing **explainability** in multi-agent decisions.
* Expanding multilingual support to cover **22 Indian languages**.

---
