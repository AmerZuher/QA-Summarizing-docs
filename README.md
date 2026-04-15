# QA-Summarizing-Docs

<div align="center">

![NLP](https://img.shields.io/badge/NLP-Natural%20Language%20Processing-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A comprehensive Natural Language Processing (NLP) toolkit for extractive and abstractive summarization, semantic question answering, grammatical analysis, and word embeddings exploration.

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Project Structure](#project-structure) • [Technologies](#technologies)

</div>

---

## 📋 Overview

QA-Summarizing-Docs is an advanced NLP project designed to demonstrate state-of-the-art techniques for document understanding, text summarization, and intelligent question answering. The project combines multiple transformer-based models and linguistic analysis tools to provide comprehensive document analysis capabilities.

This toolkit is ideal for:

- 📄 **Document Summarization**: Automatically generate concise summaries from lengthy documents
- ❓ **Question Answering**: Extract answers to questions from document context using semantic similarity
- ✏️ **Grammar Validation**: Detect and correct grammatical errors in text
- 🔤 **Semantic Analysis**: Analyze word embeddings and semantic relationships
- 🧠 **NLP Experimentation**: Explore various NLP models and techniques

---

## ✨ Features

### 1. **Abstractive Summarization with Pegasus**

- Uses Google's Pegasus model for high-quality abstractive text summarization
- Configurable summary length (30-40% of original text)
- Intelligent token management for optimal results
- Generates coherent, human-readable summaries

### 2. **Semantic Question Answering System**

- Leverages spaCy's medium-sized NLP model for semantic understanding
- Implements similarity-based answer extraction
- Customizable confidence thresholds
- Returns relevant sentences from document context

### 3. **Grammar & Syntax Checking**

- Uses LanguageTool for comprehensive grammar validation
- Identifies grammatical errors, typos, and stylistic issues
- Provides correction suggestions for detected errors
- Support for multiple languages (English, Spanish, French, etc.)

### 4. **Word Embeddings & Similarity Analysis**

- Explores word vector representations using spaCy models
- Computes semantic similarity between words and sentences
- Identifies semantically related words using cosine similarity
- Demonstrates clustering of similar concepts

### 5. **Multiple spaCy Models Support**

- **en_core_web_sm**: Small, lightweight model for faster processing
- **en_core_web_md**: Medium model with word vectors for semantic analysis

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- GPU support recommended for faster model inference (optional)

### Step 1: Clone or Download the Repository

```bash
git clone <repository-url>
cd QA-Summarizing-Docs
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Core Dependencies:**

```
transformers>=4.20.0
torch>=1.9.0
spacy>=3.0.0
language-tool-python>=2.6
jupyter>=1.0.0
numpy>=1.21.0
```

### Step 4: Download spaCy Models

```bash
# Medium model (includes word vectors - recommended for full features)
python -m spacy download en_core_web_md

# Small model (lightweight alternative)
python -m spacy download en_core_web_sm
```

### Step 5: Download Pegasus Model

The Pegasus model will be automatically downloaded on first use (~2.5GB), or manually download:

```bash
python -c "from transformers import PegasusForConditionalGeneration, PegasusTokenizer; \
          PegasusTokenizer.from_pretrained('google/pegasus-xsum'); \
          PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')"
```

---

## 📖 Usage

### Quick Start with Jupyter Notebooks

Launch Jupyter and open any notebook:

```bash
jupyter notebook
```

### 1. Text Summarization (Pegasus)

**File**: `Pegasus.ipynb`

```python
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# Load model and tokenizer
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

# Load and tokenize text
with open("your_document.txt", "r") as f:
    text = f.read()

tokens = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")

# Generate summary (30-40% of original length)
max_length = int(len(tokens.input_ids[0]) * 0.4)
min_length = int(len(tokens.input_ids[0]) * 0.3)
summary = model.generate(**tokens, max_length=max_length, min_length=min_length)

# Decode and display
summary_text = tokenizer.decode(summary[0], skip_special_tokens=True)
print(summary_text)
```

### 2. Question Answering System

**File**: `SpacyMD.ipynb`

```python
import spacy

# Load model
nlp = spacy.load('en_core_web_md')

# Process document and question
with open("document.txt", "r") as f:
    text = f.read()

doc = nlp(text)
question = "What is the main topic?"
question_doc = nlp(question)

# Find similar sentence (answer)
threshold = 0.8
answer = None

for sentence in doc.sents:
    if question_doc.similarity(sentence) > threshold:
        answer = sentence.text
        break

print(f"Answer: {answer if answer else 'No answer found'}")
```

### 3. Grammar Checking

**File**: `language_tool_python.ipynb`

```python
import language_tool_python

# Initialize grammar checker
tool = language_tool_python.LanguageTool(language='en-US')

# Read and check text
with open("document.txt", "r") as f:
    text = f.read()

matches = tool.check(text)

# Display errors and suggestions
for match in matches:
    print(f"Error: {match.ruleId}")
    print(f"Message: {match.message}")
    print(f"Suggestions: {match.replacements}")
    print("-" * 40)
```

### 4. Word Embeddings Analysis

**File**: `word embeddings.ipynb` and `SpacyMD.ipynb`

```python
import spacy
import numpy as np

nlp = spacy.load("en_core_web_md")

# Word similarity
word = "pet"
similar = nlp.vocab.vectors.most_similar(
    np.asarray([nlp.vocab.vectors[nlp.vocab.strings[word]]]), n=5
)
similar_words = [nlp.vocab.strings[w] for w in similar[0][0]]
print(f"Words similar to '{word}': {similar_words}")

# Sentence similarity
sent1 = nlp("I enjoy oranges")
sent2 = nlp("I enjoy apples")
similarity = round(sent1.similarity(sent2), 3) * 100
print(f"Similarity: {similarity}%")
```

### 5. Comparing spaCy Models

**Files**: `SpacySM.ipynb` (small) vs `SpacyMD.ipynb` (medium)

```python
import spacy

# Small model (faster, no word vectors)
nlp_sm = spacy.load('en_core_web_sm')

# Medium model (slower, with word vectors for semantic analysis)
nlp_md = spacy.load('en_core_web_md')

# Note: Semantic similarity requires word vectors (medium model)
# Small model is suitable for tokenization, POS tagging, NER only
```

---

## 📁 Project Structure

```
QA-Summarizing-Docs/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── Pegasus.ipynb                      # Abstractive summarization
├── SpacyMD.ipynb                      # QA system with medium model
├── SpacySM.ipynb                      # Alternative small model
├── language_tool_python.ipynb         # Grammar checking
├── word embeddings.ipynb              # Word vector analysis
│
├── DS_Mariana_Trench.txt             # Sample data (Wikipedia)
├── DS_MarianaTrench.txt              # Sample data variant
│
└── NLP QA & Summarizing/             # Additional resources
```

---

## 🛠️ Technologies & Models

| Component         | Technology                | Model             | Purpose                             |
| ----------------- | ------------------------- | ----------------- | ----------------------------------- |
| **Summarization** | Hugging Face Transformers | Google Pegasus    | Abstractive text summarization      |
| **NLP Pipeline**  | spaCy                     | en_core_web_md/sm | Tokenization, POS, NER, embeddings  |
| **Grammar Check** | LanguageTool              | LT v6.0+          | Grammar, spelling, style validation |
| **Deep Learning** | PyTorch                   | -                 | Model inference backend             |
| **Notebooks**     | Jupyter                   | -                 | Interactive development environment |

### Model Specifications

**Pegasus (google/pegasus-xsum)**

- Fine-tuned on XSum dataset
- 568M parameters
- Optimized for news/article summarization
- Model size: ~2.5GB

**spaCy en_core_web_md**

- 41MB model size
- Includes word vectors (300-dimensional)
- 96.89% accuracy on POS tagging
- Supports 17 entity types

**spaCy en_core_web_sm**

- 13MB model size (lightweight)
- No word vectors
- Faster inference
- Suitable for tokenization and basic NLP

---

## 📊 Example Workflow

### Complete Document Analysis Pipeline

```python
import spacy
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import language_tool_python

# 1. Load models
nlp = spacy.load('en_core_web_md')
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
grammar_tool = language_tool_python.LanguageTool(language='en-US')

# 2. Load document
with open("document.txt", "r") as f:
    text = f.read()

# 3. Grammar check
matches = grammar_tool.check(text)
print(f"Found {len(matches)} grammar issues")

# 4. Summarize
tokens = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
summary = model.generate(**tokens, max_length=int(len(tokens.input_ids[0])*0.4))
summary_text = tokenizer.decode(summary[0], skip_special_tokens=True)
print(f"Summary: {summary_text}")

# 5. Answer question
doc = nlp(text)
question = "What is the main topic?"
question_doc = nlp(question)

for sentence in doc.sents:
    if question_doc.similarity(sentence) > 0.8:
        print(f"Answer: {sentence.text}")
        break

# 6. Extract entities
for ent in doc.ents:
    print(f"{ent.text} - {ent.label_}")
```

---

## 🎯 Use Cases

- **Document Management**: Automatically summarize large document collections
- **Customer Support**: Build QA systems for FAQ automation
- **Content Moderation**: Detect and flag low-quality or grammatically incorrect content
- **Semantic Search**: Find semantically similar documents and passages
- **Research**: Explore NLP techniques and model architectures
- **Data Preprocessing**: Clean and validate text data for ML pipelines

---

## ⚙️ Configuration & Customization

### Summarization Parameters

```python
# Adjust summary length
min_percentage = 0.25  # 25% of original
max_percentage = 0.50  # 50% of original

# Token truncation strategies
truncation=True, padding="longest"  # Recommended for Pegasus
```

### Question Answering Threshold

```python
# Lower threshold = more results but potentially less relevant
threshold = 0.7  # Standard setting
threshold = 0.8  # Strict matching
threshold = 0.6  # Loose matching
```

### spaCy Model Selection

```python
# Speed vs. Accuracy tradeoff
nlp = spacy.load('en_core_web_sm')  # Fast, no semantics
nlp = spacy.load('en_core_web_md')  # Balanced
nlp = spacy.load('en_core_web_lg')  # Slow, best accuracy
```

---

## 📈 Performance Metrics

| Model        | Speed  | Memory | Features          | Best For            |
| ------------ | ------ | ------ | ----------------- | ------------------- |
| Pegasus      | Medium | 2.5GB  | Summarization     | News, articles      |
| spaCy-SM     | Fast   | 13MB   | Tokenization, POS | Speed-critical apps |
| spaCy-MD     | Normal | 41MB   | + Word vectors    | Semantic tasks      |
| LanguageTool | Slow   | ~100MB | Grammar, spelling | Quality assurance   |

**Estimated Processing Time** (on CPU):

- Summarization: 2-5 seconds per page
- QA extraction: 0.1-0.5 seconds per document
- Grammar check: 1-2 seconds per page
- Word similarity: <0.1 seconds

---

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:

- [ ] Add more language support
- [ ] Implement fine-tuning scripts
- [ ] Add API wrapper for production deployment
- [ ] Expand test coverage
- [ ] Optimize inference performance
- [ ] Add GPU support documentation

---

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🔗 Resources & References

### Research Papers

- **Pegasus**: [PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://arxiv.org/abs/1912.08777)
- **spaCy**: [spaCy: Industrial-strength Natural Language Processing](https://spacy.io/)

### Documentation

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [spaCy Official Documentation](https://spacy.io/api)
- [LanguageTool API](https://www.languagetoolplus.com/api)
- [PyTorch Documentation](https://pytorch.org/docs)

### Related Tools

- [Hugging Face Model Hub](https://huggingface.co/models)
- [spaCy Universe](https://spacy.io/universe)
- [Jupyter Documentation](https://jupyter.org/)

---

## ❓ FAQ

**Q: Do I need a GPU to run this?**
A: GPU is optional but recommended for faster inference. CPU mode works but is slower.

**Q: Can I use this for production?**
A: Yes, with optimization. Consider:

- Model quantization for smaller size
- Batch processing for throughput
- API wrapper (Flask/FastAPI) for deployment
- Caching for repeated queries

**Q: Why is the Pegasus model so large?**
A: 2.5GB includes 568M parameters fine-tuned on XSum dataset. It's among the best abstractive summarization models.

**Q: Can I fine-tune these models on custom data?**
A: Yes! Pegasus and spaCy models can be fine-tuned. See their respective documentation.

**Q: What's the difference between spaCy models?**
A: Size/speed vs. accuracy tradeoff. MD includes word vectors for semantic similarity; SM is faster but lacks semantic features.

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'spacy'`

```bash
pip install spacy
python -m spacy download en_core_web_md
```

**Issue**: CUDA out of memory

```python
# Use CPU instead
import torch
torch.cuda.is_available()  # Check GPU
# Set device to CPU in model configs
```

**Issue**: Very slow inference

- Use smaller model (spaCy-SM)
- Enable GPU acceleration
- Use batch processing
- Reduce input size

---

## 📞 Support & Contact

For issues, questions, or suggestions:

- 🐙 GitHub Issues: [Report a bug]
- 💬 Discussions: [Ask questions]
- 📧 Email: [Contact]

---

## 🙏 Acknowledgments

- Google Research for Pegasus model
- Explosion AI for spaCy framework
- LanguageTool community
- Hugging Face for transformers library
