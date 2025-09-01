# LLM-Powered Data Analysis

Natural language data analysis powered by LLMs – ask questions about your JSON datasets in plain English and get intelligent analysis results.

---

## Description

**Generalized RAG Dataset Chat** leverages Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs) to interpret arbitrary user queries and automatically generate, execute, and summarize Python data analysis on any loaded dataset.

---

## Features

- **Retrieval-Augmented Generation (RAG):** Uses an LLM to interpret user intent and augment queries with relevant context from your dataset.
- **Automatic Schema Analysis:** Loads and analyzes the structure of your JSON dataset to provide context-aware answers.
- **LLM-Driven Code Generation:** Dynamically generates Python code to analyze data based on natural language queries.
- **Secure Code Execution:** Runs the generated analysis code on your dataset and safely formats the results.
- **Natural Language Responses:** Returns analysis results in clear, human-readable language.
- **Robust Error Handling:** Includes fallback strategies to handle ambiguous queries or code generation failures.
- **Flexible Query Support:** Handles filtering, grouping, counting, summarization, and more.

---

## How It Works

The core class, `GeneralizedRAGChat`, provides a seamless interface for:
1. **Loading a JSON dataset:** The structure is automatically analyzed for efficient querying.
2. **Understanding user queries:** Natural language queries are parsed and interpreted using an LLM.
3. **Generating Python code:** The LLM produces relevant Python code to perform the requested analysis.
4. **Executing analysis:** Generated code is executed on the dataset.
5. **Formatting responses:** Results are summarized and returned in plain English.

---

## Getting Started

### Prerequisites

- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/)
- An OpenAI API key (set as the environment variable `OPENAI_API_KEY`)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/GabrielSantiniFrancisco/llm-powered-data-analysis.git
   cd llm-powered-data-analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

Edit your configuration in `conf/RagChatSession.cfg`:
- Set model, API key, logging, and paths as needed.
- Example environment variable setup:
  ```bash
  export OPENAI_API_KEY=your_api_key_here
  ```

### Usage

1. **Prepare your JSON dataset.**
   Place your dataset, e.g., `data.json`, in the project directory.

2. **Run the main script:**
   ```bash
   python main.py --data data.json
   ```

3. **Ask questions about your data:**
   - "How many users signed up in August?"
   - "Summarize sales by region."
   - "Show the average score for group B."

4. **Receive natural language answers and actionable analysis!**

---

## Example

```bash
$ python main.py --data employees.json
> What is the average age of employees in the Engineering department?
Answer: The average age of employees in the Engineering department is 36.7 years.
```

---

## Contributing

Contributions are welcome! Please open issues or pull requests to suggest features, report bugs, or improve the documentation.

---

## License

This project is licensed under the MIT License.

---

*Empowering data analysis through natural language and AI!*
