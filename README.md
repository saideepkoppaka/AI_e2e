# AI_e2e: End-to-End AI Fundamentals

This repository features a series of Jupyter notebooks designed to help you master Python, data science, statistics, machine learning, deep learning, NLP, RAG, and AI Agents with a special focus on generative AI.

---

## 🚀 Getting Started: Running the Notebooks

### 1. Clone the Repository

```bash
git clone https://github.com/saideepkoppaka/AI_e2e.git
cd AI_e2e
```

### 2. Create and Activate a Virtual Environment

#### On **Linux/MacOS**:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### On **Windows**:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Lab

```bash
jupyter lab
```

Your browser will open Jupyter Lab. Open any `.ipynb` file and run the cells!

---

## 📚 Notebook Summaries

### 1. `python_start.ipynb`
**Summary:**  
A gentle introduction to Python for AI. Covers:
- Why Python is essential for AI/ML
- Basic syntax, print statements
- Imports, standard library usage
- Platform-agnostic features

---

### 2. `ML_Statistics_Notebook.ipynb`
**Summary:**  
Core statistics for machine learning, including:
- Mean, median, mode, variance, standard deviation
- Probability basics
- Normal distribution

---

### 3. `pandas_fundamentals_with_visuals.ipynb`
**Summary:**  
Hands-on pandas with a sample employee dataset:
- DataFrame creation and manipulation
- Visualizations with matplotlib
- Advanced pandas operations (reshaping, rolling windows)

---

### 4. `Logistic_Regression_Tutorial.ipynb`
**Summary:**  
A practical walkthrough of logistic regression using scikit-learn:
- Data loading and preprocessing
- Model training and evaluation
- Hyperparameter tuning and metrics (accuracy, ROC, confusion matrix)
- Visualization with seaborn/matplotlib

---

### 5. `PyTorch_DeepLearning_Intro.ipynb`
**Summary:**  
Step-by-step deep learning in PyTorch:
- Neural network from scratch
- Tensors, DataLoader, model layers, activation
- Training loop, loss, optimizer
- Regularization & evaluation using the Iris dataset

---

### 6. `Next_Word_Prediction_Evolution.ipynb`
**Summary:**  
Evolution of next-word prediction in NLP:
- Basic Python approaches
- Regex-based patterns
- N-gram language model
- Stepwise code explanations and results

---

### 7. `RNN_deepdive.ipynb`
**Summary:**  
A hands-on exploration of RNNs:
- Data preparation for NLP
- Custom RNN implementation using PyTorch
- Sequence modeling for next-word prediction

---

### 8. `Transformer_mini.ipynb`
**Summary:**  
Building a minimal transformer model:
- Explanation of attention mechanism
- Creating vocab, encoding, and model logic from scratch
- Stepwise implementation and experiment

---

### 9. `Transformer_final.ipynb`
**Summary:**  
Full transformer implementation:
- Decoder-only transformer with causal masking
- Data prep with <sos>/<eos> tokens
- Attention mask, inference, and advanced logic

---
## 📁 Directory Structure

- `collab_notebooks/`: Contains Google Colaboratory notebooks. These are self-contained and may install their own dependencies inside the notebook.
- Other `.ipynb` files at the root or in other directories: Designed to be run locally using the repo's `requirements.txt`.

---

### Notebooks in `collab_notebooks/` (Colab-ready)

#### 1. `Finetuning_101.ipynb`
**Purpose:**  
Covers the basics of fine-tuning machine learning models, likely including how to adapt a pre-trained model to a custom dataset using modern libraries (such as Hugging Face or PyTorch).  
**Features:**  
- Step-by-step walkthrough of the fine-tuning process
- Includes code cells to install required packages directly in Colab

#### 2. `Hugging_face_101.ipynb`
**Purpose:**  
An introduction to the Hugging Face ecosystem, focusing on loading and using LLM models.  
**Features:**  
- Demonstrates use of the `transformers` library
- May cover pipelines for text classification, generation, etc.

#### 3. `Langgraph_basic.ipynb`
**Purpose:**  
An introduction to LangGraph, likely walking through basic graph-based language modeling or knowledge graph techniques.  
**Features:**  
- Installation and usage of relevant libraries inside Colab
- Example workflows or pipelines

#### 4. `RAG.ipynb`
**Purpose:**  
Covers Retrieval-Augmented Generation (RAG), combining retrieval models with generative models for enhanced QA or knowledge-based tasks.  
**Features:**  
- Demonstrates how to integrate an external knowledge base with a language model
- Shows installation of dependencies within the notebook

#### 5. `langchain_prompt_engineering.ipynb`
**Purpose:**  
A practical guide to prompt engineering using the LangChain framework, optimizing prompts for various LLM tasks.  
**Features:**  
- Shows how to use LangChain tools for prompt design and evaluation
- Installs its own dependencies in Colab

#### 6. `Langgraph_agent_with_tools_py.ipynb`
**Purpose:** 
Showcases a modular, multi-agent Stack Overflow-style assistant that answers Python and data engineering queries using a combination of custom knowledge base retrieval, real-time web search, and LLM summarization, all orchestrated via LangGraph.

**Features:** 
- Integrates Tavily Search API for up-to-date web answers
- Uses a local FAISS vector store for fast retrieval from a Stack Overflow-style KB
- Leverages OpenAI GPT-4o for answer synthesis and summarization
- Clearly separates “tools” (API/function calls) and “agents” (decision logic)
- Asynchronous execution for efficient API use
- Automatically logs unanswered/uncertain questions for review

**System Workflow:**
- Knowledge Base Agent tries to answer from the KB
- If uncertain, triggers a Web Search Agent
- Summarization Agent combines and refines all results using LLMs
- Logger Agent stores challenging queries for future improvement


## ✨ Author

Maintained by [saideepkoppaka](https://github.com/saideepkoppaka)
