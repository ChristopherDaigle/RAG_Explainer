# RAG System Educational Notebook
**University of Connecticut - Masters of Science in Quantitative Economics (MSQE)**

## Overview

This interactive Jupyter notebook teaches you how to build a **Retrieval-Augmented Generation (RAG)** system from scratch. You'll learn how modern AI systems combine document retrieval with language models to provide accurate, grounded responses.

### What You'll Learn

- 📚 **Document Processing**: How to chunk large documents for efficient retrieval
- 🧮 **Embeddings**: Converting text into semantic vector representations
- 🔍 **Similarity Search**: Finding relevant information using cosine similarity
- 📊 **Visualization**: Understanding embedding spaces with UMAP dimensionality reduction
- 🤖 **RAG Pipeline**: Constructing prompts for Large Language Models (LLMs)

### Why RAG Matters

RAG systems are revolutionizing how we interact with information:
- **Reduces AI hallucinations** by grounding responses in actual documents
- **Enables domain-specific AI** without retraining models
- **Powers modern applications** like chatbots, research assistants, and Q&A systems
- **Critical skill** for data science and AI engineering careers

---

## Prerequisites

### Knowledge Requirements
- Basic Python programming
- Familiarity with pandas and numpy
- Understanding of basic machine learning concepts (helpful but not required)

### Technical Requirements
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB, 8GB+ recommended
- **Storage**: ~2GB for dependencies
- **OS**: macOS, Linux, or Windows

---

## Getting Started

### Step 1: Set Up Your Environment

Create and activate a new conda environment specifically for this project:
```bash
# Create a new conda environment
conda create --name rag_msqe python=3.11

# Activate the environment
conda activate rag_msqe

# Install required packages
pip install -r requirements.txt

# Install Jupyter kernel for this environment
python -m ipykernel install --user --name=rag_msqe
```

**Why a separate environment?**
- Prevents dependency conflicts with other projects
- Ensures reproducibility across different machines
- Makes it easy to share and deploy

### Step 2: Launch Jupyter
```bash
# Make sure your environment is activated
conda activate rag_msqe

# Launch Jupyter Lab (recommended) or Jupyter Notebook
jupyter lab
# OR
jupyter notebook
```

### Step 3: Select the Correct Kernel

1. Open `rag_tutorial.ipynb`
2. In the top-right corner, click on the kernel name
3. Select **"rag_msqe"** from the dropdown
4. If you don't see it, restart Jupyter and try again

### Step 4: Run the Notebook

Start with the first cell and run sequentially:
- **Shift + Enter**: Run cell and move to next
- **Ctrl/Cmd + Enter**: Run cell and stay
- **Kernel → Restart & Run All**: Run entire notebook

⚠️ **Important**: The first run will download embedding models (~100MB), which may take a few minutes.

---

## Notebook Structure

### Section 0: Setup & Dependencies
- Install and verify all required libraries
- Check for GPU availability (optional but faster)

### Section 1: Document Ingestion
- Load sample documents (Economics, Literature, Philosophy)
- Understand document characteristics

### Section 2: Document Chunking
- Learn why chunking is necessary
- Experiment with different chunk sizes and overlap

### Section 3: Chunk Embedding
- Generate semantic embeddings using sentence transformers
- Visualize individual embedding vectors

### Section 4: Chunk Storage
- Store embeddings in pandas DataFrames
- Understand similarity distributions
- Visualize embeddings in 2D and 3D

### Section 5: User Query Embedding
- Create queries and embed them
- Preview similarity scores

### Section 6: Chunk Retrieval
- Implement top-k retrieval
- Implement threshold-based retrieval
- Compare retrieval methods

### Section 7: Prompt Construction
- Build complete RAG prompts
- Understand prompt engineering

### Section 8: Experimentation
- Guided experiments with different configurations
- Analyze trade-offs

---

## Key Concepts Covered

### 1. Embeddings
Dense vector representations that capture semantic meaning. Similar concepts have similar vectors.

### 2. Cosine Similarity
Measures the angle between two vectors to determine semantic similarity:
```
similarity = (A · B) / (||A|| ||B||)
```
Range: -1 (opposite) to 1 (identical)

### 3. UMAP (Uniform Manifold Approximation and Projection)
Non-linear dimensionality reduction for visualizing high-dimensional embeddings in 2D/3D.

### 4. Top-K vs Threshold Retrieval
- **Top-K**: Retrieve exactly k most similar chunks
- **Threshold**: Retrieve all chunks above similarity threshold
- **Combined**: Best of both worlds

---

## Troubleshooting

### NumPy Compatibility Issues

If you see errors about NumPy 2.x incompatibility:
```bash
pip install "numpy<2.0"
```

Then restart your Jupyter kernel (Kernel → Restart).

### Tokenizers Warning

If you see warnings about tokenizers parallelism, add this to the first cell:
```python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

### Out of Memory

If you run out of memory during UMAP visualization:
- Reduce the number of chunks to visualize (modify `NUM_CHUNKS_TO_PLOT`)
- Use the smaller embedding model (`all-MiniLM-L6-v2`)
- Close other applications

### Slow Embedding Generation

- **With GPU**: ~30-60 seconds for 100 chunks
- **With CPU**: ~2-5 minutes for 100 chunks

If it's too slow, use the smaller model:
```python
MODEL_NAME = "all-MiniLM-L6-v2"
```

### Kernel Dies or Crashes

1. Restart the kernel: **Kernel → Restart**
2. Clear outputs: **Kernel → Restart & Clear Output**
3. If issues persist, recreate the environment:
```bash
   conda deactivate
   conda remove --name rag_msqe --all
   # Then follow setup steps again
```

---

## Experimentation Guide

### Try Different Configurations

**Chunk Sizes:**
```python
CHUNK_SIZE = 200   # More precise, less context
CHUNK_SIZE = 500   # Balanced (default)
CHUNK_SIZE = 1000  # More context, less precise
```

**Retrieval Parameters:**
```python
TOP_K = 3              # Fewer chunks
TOP_K = 10             # More comprehensive

SIMILARITY_THRESHOLD = 0.2   # More permissive
SIMILARITY_THRESHOLD = 0.5   # More strict
```

**Embedding Models:**
```python
MODEL_NAME = "all-MiniLM-L6-v2"    # Fast (384 dimensions)
MODEL_NAME = "all-mpnet-base-v2"   # Better quality (768 dimensions)
```

### Sample Queries to Try

**Economics:**
- "How do supply and demand determine market prices?"
- "What are the different types of unemployment?"
- "Explain monetary policy and central banks"

**Literature:**
- "What is the difference between Romanticism and Realism?"
- "How do authors use symbolism in literature?"
- "Explain narrative point of view"

**Philosophy:**
- "What is the mind-body problem?"
- "Explain rationalism versus empiricism"
- "What is Kant's categorical imperative?"

---

## Expected Outcomes

By the end of this notebook, you will:

✅ Understand how RAG systems work end-to-end  
✅ Be able to implement basic semantic search  
✅ Understand embeddings and similarity metrics  
✅ Know how to evaluate and optimize retrieval quality  
✅ Be prepared to build production RAG systems  

---

## Advanced Topics & Next Steps

### Production Considerations

This notebook uses DataFrames for simplicity. Production systems use:

- **Vector Databases**: Pinecone, Weaviate, Chroma, Qdrant
- **Reranking Models**: Cross-encoders for better accuracy
- **Hybrid Search**: Combine keyword and semantic search
- **Metadata Filtering**: Filter by date, source, author, etc.

### Further Learning Resources

**Documentation:**
- [Sentence Transformers](https://www.sbert.net/)
- [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)
- [Pinecone Learning Center](https://www.pinecone.io/learn/)

---

## Course Information

**Program**: Masters of Science in Quantitative Economics (MSQE)  
**Institution**: University of Connecticut  
**Topic**: Retrieval-Augmented Generation Systems

### Learning Objectives

1. Understand the theoretical foundations of RAG systems
2. Implement document retrieval using semantic search
3. Evaluate trade-offs in chunking and retrieval strategies
4. Design prompts for grounded language model generation
5. Apply RAG techniques to real-world problems

---

## Support & Questions

### During Class
- Ask your instructor for clarification
- Work with classmates to debug issues
- Use the experimentation section to explore

### Outside Class
- Review the notebook's markdown explanations
- Check the troubleshooting section
- Consult the documentation links provided

---

## File Structure
```
.
├── README.md                 # This file
├── requirements.txt          # Python dependencies
└── rag_tutorial.ipynb        # Main educational notebook
```

---

## Dependencies

All dependencies are specified in `requirements.txt`:

- **Core Libraries**: pandas, numpy, scikit-learn
- **Visualization**: matplotlib, seaborn, plotly
- **ML/AI**: sentence-transformers, transformers, torch
- **Dimensionality Reduction**: umap-learn
- **Development**: ipykernel, ipywidgets, tqdm

### Version Notes

- Using NumPy 2.3.4 (may need to downgrade to <2.0 for compatibility)
- PyTorch 2.9.0+ for optimal performance
- sentence-transformers 5.1.1+ for latest features

---

**Ready to build your first RAG system? Open `rag_tutorial.ipynb` and let's get started! 🚀**