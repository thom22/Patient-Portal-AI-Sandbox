# Secure AI Sandbox for Patient Portal Messaging

> **Companion code for:**  
> Gleason KT, Kidu T, Babu V, Hasselfled B, & Wolff JL. *A Secure User Interface for Pre-Clinical Evaluation of Artificial Intelligence in Patient-Portal Message Management: A Tutorial.*

A secure, interactive platform for testing AI use cases on patient portal messages with built-in de-identification pipeline. The sandbox enables clinical and technical teams to experiment with LLM-powered workflows on de-identified data **before** clinical integration.

---

## Features

- **De-Identification Pipeline** — Multi-method PHI detection (spaCy NER, scispaCy biomedical NER, regex) with medication preservation and synthetic replacement  
- **4 AI Use Cases** — Authorship detection, criticality flagging, message categorization, and draft response generation  
- **Prompt Studio** — View, edit, and experiment with prompt templates at runtime  
- **Dual Mode** — Runs in simulation (mock AI, zero setup) or live (Azure OpenAI)  
- **Batch Processing** — Upload CSV files for bulk analysis with progress tracking  
- **Cost Dashboard** — Token-level cost telemetry per message and per use case  

---

## Quick Start

```bash
# 1. Clone the repository
git clone <repository-url>
cd <repository-name>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install spaCy models
python -m spacy download en_core_web_sm
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

# 4. Run (simulation mode — no API key needed)
streamlit run app.py
```

---

## Project Structure

```
Patient-Portal-AI-Sandbox/
├── app.py                      # Streamlit application (6 pages)
├── ai_framework.py             # AI use case registry, prompts, factory
├── deidentification_engine.py  # PHI detection & replacement engine
├── config.py                   # Configuration (auto-loads .env)
├── synthetic_data.py           # Synthetic message generator
├── requirements.txt            # Python dependencies
├── .env.example                # Configuration template
├── .gitignore
├── LICENSE
├── data/
│   ├── synthetic_messages.csv  # Auto-generated synthetic dataset
│   └── sample_messages.csv     # 20 diverse test messages
└── saved_scenarios/            # User-saved scenario outputs
```

---

## Application Pages

| Page | Description |
|------|-------------|
| **Home** | Mode status, dataset overview, quick navigation |
| **Dataset Explorer** | Browse messages by Enterprise ID, side-by-side conventional vs AI-augmented views |
| **Manual Input** | Paste a single message → de-identify → run AI analysis step-by-step |
| **Batch Upload** | Upload CSV/Excel → full pipeline → summary dashboard → download results |
| **Prompt Studio** | View and edit prompt templates for each AI use case |
| **Cost Dashboard** | Token usage and cost analysis across all sessions |

---

## Connecting to Azure OpenAI (Live Mode)

1. Copy the configuration template:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your credentials:
   ```
   SMARTREP_MODE=live
   AZURE_OPENAI_API_KEY=your-api-key
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_MODEL=gpt-4
   USE_SPACY=true
   USE_SCISPACY=true
   ```

3. Run:
   ```bash
   streamlit run app.py
   ```

Switch back to simulation mode anytime by changing `SMARTREP_MODE=simulation`.

---

## Citation

If you use this software in your research, please cite:

```
Gleason, K.T., Kidu, T., Babu, V., Hasselfled, B., & Wolff, J.L.
A Secure User Interface for Pre-Clinical Evaluation of Artificial Intelligence in Patient-Portal Message Management: A Tutorial.
```

---

## License

This project is released for academic and research purposes under the [MIT License](LICENSE).