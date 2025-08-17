# 🏛️ NitiAI - Government Scheme Assistant

NitiAI is an AI-powered assistant that helps Indian citizens discover and apply for government welfare schemes. Built using LangChain, Groq LLaMA 3 (70B), and ChromaDB, it provides personalized scheme recommendations based on user demographics.

![NitiAI Demo](https://via.placeholder.com/800x400?text=NitiAI+Demo)

## 🌟 Features

- **🔍 Smart Scheme Discovery**: Find relevant government schemes based on your profile
- **✅ Eligibility Checking**: Check if you're eligible for specific schemes
- **📋 Document Requirements**: Get detailed information about required documents
- **💰 Benefits Information**: Learn about financial benefits and assistance
- **🤖 AI-Powered Chat**: Natural language interaction with the assistant
- **📊 Personalized Recommendations**: Tailored suggestions based on age, income, location, and category

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI Model**: Groq LLaMA 3 (70B)
- **Framework**: LangChain
- **Vector Database**: ChromaDB
- **Embeddings**: HuggingFace Sentence Transformers
- **Summarization**: Facebook BART
- **Language**: Python 3.8+

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Groq API key ([Get it here](https://console.groq.com/))
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/nitiai.git
   cd nitiai
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API key**
   
   Create `.streamlit/secrets.toml` file:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open in browser**
   Navigate to `http://localhost:8501`

## 📁 Project Structure

```
NITIAI/
├── app.py                      # Main Streamlit application
├── utils/
│   ├── __init__.py
│   ├── user_profile.py         # User profile management
│   ├── tools.py               # LangChain tools
│   └── agent_setup.py         # Agent configuration
├── chroma_store/              # Vector database (ChromaDB)
├── data/
│   ├── policy_data.csv        # Original policy data
│   └── updated_dataset.csv    # Processed dataset
├── notebooks/
│   └── data_preprocessing.ipynb # Data processing notebook
├── requirements.txt           # Python dependencies
├── .streamlit/
│   └── secrets.toml          # API keys (not in git)
├── .gitignore
└── README.md
```

## 🔧 Configuration

### Environment Variables

- `GROQ_API_KEY`: Your Groq API key for accessing LLaMA 3

### Streamlit Secrets

Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your_api_key_here"
```

## 🎯 Usage

1. **Complete Your Profile**: Fill in your age, income, location, and category in the sidebar
2. **Ask Questions**: Use natural language to ask about government schemes
3. **Get Recommendations**: Receive personalized scheme suggestions
4. **Check Eligibility**: Verify if you're eligible for specific schemes
5. **Get Documents**: Find out what documents you need to apply

### Example Queries

- "What schemes can I apply for?"
- "Show me employment schemes for students"
- "Am I eligible for PM Kisan scheme?"
- "What documents do I need for Mudra Loan?"
- "Explain the startup India scheme"

## 🏗️ System Architecture

```
User Input → Streamlit UI → LangChain Agent → Tools → ChromaDB → Groq LLaMA → Response
```

### Key Components

1. **UserProfile**: Manages user demographic information
2. **Tools**: 5 specialized tools for different tasks
   - SchemeSearchTool
   - EligibilityCheckTool
   - DocumentRequirementTool
   - BenefitsSearchTool
   - SchemeSummaryTool
3. **Agent**: ReAct agent that decides which tool to use
4. **VectorStore**: ChromaDB for semantic search

## 🚀 Deployment

### Streamlit Cloud

1. Fork this repository
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Deploy from your GitHub repository
4. Set `GROQ_API_KEY` in the secrets section

### Local Docker (Optional)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

## 📊 Data Sources

The system uses government scheme data including:
- Central Government Schemes
- State Government Schemes
- Eligibility Criteria
- Application Processes
- Required Documents
- Benefits Information

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and commit: `git commit -m 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black .
isort .
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔍 Troubleshooting

### Common Issues

1. **API Key Error**
   - Ensure your Groq API key is correctly set in secrets.toml
   - Check if the API key has proper permissions

2. **ChromaDB Issues**
   - Delete the `chroma_store` folder and restart the app to rebuild the database

3. **Memory Issues**
   - The app loads large language models; ensure you have sufficient RAM (8GB+ recommended)

## 📞 Discussions
<!-- 
- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/nitiai/issues) -->
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/nitiai/discussions)

## 🙏 Acknowledgments

- Groq for providing fast LLaMA inference
- LangChain for the agent framework
- Streamlit for the web interface
- HuggingFace for embeddings and models
- Government of India for open data access

## 📈 Roadmap

- [ ] Add more regional languages
- [ ] Include more government schemes
- [ ] Add application status tracking
- [ ] Mobile app version
- [ ] Integration with government portals
- [ ] Voice interface
- [ ] Document upload and verification

---

**Made with ❤️ for Indian Citizens**