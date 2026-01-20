## 🖼️ Giving Eyes to Your AI: Engineering a Multimodal Agent

A hands-on workshop exploring multimodal AI agents with [Haystack](https://haystack.deepset.ai/).

### What You'll Build
- 📄 Multimodal indexing pipeline (PDFs + images) using CLIP embeddings
- 🤖 Vision-enabled agent powered by GPT-4o
- 🔍 RAG tool for searching company policies
- 💬 Conversational memory for context-aware interactions
- 🔁 Human-in-the-loop controls for sensitive actions

### Get Started

👉 See **[multimodal_agent_notebook.ipynb](multimodal_agent_notebook.ipynb)** for the full interactive experience.

### Deploy with Hayhooks

Want to deploy the agent as an API? Check out **[multimodal-agent/pipeline_wrapper.py](multimodal-agent/pipeline_wrapper.py)** — a Python script version of the notebook with [Hayhooks](https://github.com/deepset-ai/hayhooks) integration pre-configured for serving the conversational agent.

### Files

The `files/` directory contains the sample data used in the workshop:
- `receipt.jpeg` — A sample receipt image for the expense reimbursement demo
- `social_budget_policy.md` — Company policy document for retrieval

### Requirements

- Python 3.10+
- OpenAI API key (or your preferred LLM provider)
- See the notebook for full package installation instructions
