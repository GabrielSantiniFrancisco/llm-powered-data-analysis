# Author : Gabriel Francisco
# Email  : gabrielsantinifrancisco@outlook.com

# Description:
#     Generalized RAG Dataset Chat leveraging LLMs to interpret arbitrary user queries
#     and automatically generate, execute, and summarize Python data analysis on any loaded dataset.


import os
import sys
import json
import traceback
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from CallOpenAi import OpenAIChatSession
from CustomLogger import CustomLogger
from EnvManager import EnvManager

class SecurityError(Exception):
    """Raised when generated code fails security validation."""
    pass

class GeneralizedRAGChat:
    """
    GeneralizedRAGChat provides a Retrieval-Augmented Generation (RAG) system that leverages a Large Language Model (LLM) to interpret arbitrary user queries and dynamically perform data analysis on a complete dataset.
    This class loads a dataset from a JSON file, analyzes its structure, and uses an LLM to:
    - Understand user queries
    - Generate Python code for data analysis
    - Execute the generated code on the dataset
    - Format the analysis results into natural language responses

    Key Features:
    - Automatic dataset schema analysis for context-aware query handling
    - LLM-driven code generation and result formatting
    - Security validation of generated code before execution
    - Fallback analysis for robust error handling
    - Detection of installed data analysis libraries for code generation
    - Support for queries requiring filtering, grouping, counting, and summarization

    Args:
        env (EnvManager): Environment manager for configuration and secrets.
        logger (CustomLogger): Logger for tracking events and errors.
        json_file_path (str): Path to the JSON file containing the dataset.
        context_file_path (str, optional): Path to additional context for the chat session.

    Attributes:
        env (EnvManager): Environment manager instance.
        logger (CustomLogger): Logger instance.
        complete_dataset (list): Loaded dataset as a list of records.
        chat_session (OpenAIChatSession): LLM-powered chat session for code generation and formatting.
        dataset_schema (dict): Analyzed schema of the dataset for context.

    Methods:
        send(user_query: str) -> str:
            Processes a user query by generating analysis code, executing it, and formatting the results.
        get_dataset_info() -> Dict[str, Any]:
            Returns basic information about the loaded dataset.

    Internal Methods:
        _load_complete_dataset(json_file_path: str) -> None:
            Loads the dataset from a JSON file.
        _analyze_dataset_structure() -> Dict[str, Any]:
            Analyzes the dataset to extract schema information.
        _generate_analysis_code(user_query: str) -> str:
            Uses LLM to generate Python code for data analysis.
        _validate_code_security(code_string: str) -> tuple[bool, str]:
            Validates generated code for security threats using AST parsing.
        _detect_installed_libraries() -> List[str]:
            Detects which data analysis libraries are installed in the environment.
        _execute_analysis_code(analysis_code: str) -> Dict[str, Any]:
            Executes the generated Python code on the dataset.
        _format_results_to_response(user_query: str, analysis_results: Dict[str, Any]) -> str:
            Uses LLM to format analysis results into a natural language response.
        _fallback_analysis(user_query: str) -> Dict[str, Any]:
            Performs basic analysis if code generation or execution fails.
    """
    
    def __init__(self, env: EnvManager, logger: CustomLogger, json_file_path: str, context_file_path: str = None):
        """Initialize with complete dataset and LLM understanding."""
        self.env = env
        self.logger = logger
        self.complete_dataset = []
        self._load_complete_dataset(json_file_path)
        self.chat_session = OpenAIChatSession(env, logger, context_file_path)
        self.dataset_schema = self._analyze_dataset_structure()
        self.logger.info(f"Generalized RAG chat initialized with {len(self.complete_dataset)} records")

    def _load_complete_dataset(self, json_file_path: str) -> None:
        """Load the complete dataset into memory."""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file: raw_data = json.load(file)

            if isinstance(raw_data, list): self.complete_dataset = raw_data
            elif isinstance(raw_data, dict):
                if 'data' in raw_data: self.complete_dataset = raw_data['data']
                else: self.complete_dataset = list(raw_data.values())
            else: self.complete_dataset = [raw_data]
            self.logger.info(f"Loaded {len(self.complete_dataset)} records")
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise

    def _analyze_dataset_structure(self) -> Dict[str, Any]:
        """Analyze the dataset structure to provide schema context to LLM."""
        if not self.complete_dataset: return {}

        sample_size = min(100, len(self.complete_dataset))
        sample_records = self.complete_dataset[:sample_size]

        all_fields = set()
        field_types = {}
        field_examples = {}

        for record in sample_records:
            if isinstance(record, dict):
                for field, value in record.items():
                    all_fields.add(field)
                    if field not in field_types: field_types[field] = set()
                    field_types[field].add(type(value).__name__)
                    if field not in field_examples: field_examples[field] = []
                    if len(field_examples[field]) < 5: field_examples[field].append(value)

        categorical_fields = {}
        for field in all_fields:
            values = [record.get(field) for record in sample_records if field in record]
            unique_values = list(set(values))
            if len(unique_values) <= 50: categorical_fields[field] = unique_values

        return {
            'total_records': len(self.complete_dataset),
            'fields': list(all_fields),
            'field_types': {k: list(v) for k, v in field_types.items()},
            'field_examples': field_examples,
            'categorical_fields': categorical_fields,
            'sample_records': sample_records[:3]
        }

    def send(self, user_query: str) -> str:
        """
        Process any user query by using LLM to understand intent and execute analysis.
        Step 1: Use LLM to understand the query and generate Python analysis code
        Step 2: Execute the generated analysis code on the complete dataset
        Step 3: Use LLM to format the results into natural language response
        """
        try:
            analysis_code = self._generate_analysis_code(user_query)
            analysis_results = self._execute_analysis_code(analysis_code)
            final_response = self._format_results_to_response(user_query, analysis_results)
            return final_response            
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            self.logger.debug(f'\n{traceback.format_exc()}')
            return f"I encountered an error analyzing your query: {e}"

    def _validate_code_security(self, code_string: str) -> tuple[bool, str]:
        """Analyze generated code for security threats using AST parsing."""
        import ast
        
        try:
            # Parse the code into AST
            tree = ast.parse(code_string)
        except SyntaxError as e:
            return False, f"Syntax error in generated code: {e}"
        
        # Define dangerous patterns
        dangerous_imports = {
            'os', 'sys', 'subprocess', 'shutil', 'glob', 'pathlib',
            'urllib', 'requests', 'socket', 'http', 'ftplib',
            'pickle', 'marshal', 'shelve', 'dbm',
            'ctypes', 'multiprocessing', 'threading',
            'importlib', '__builtin__', 'builtins'
        }
        
        dangerous_functions = {
            'eval', 'exec', 'compile', 'open', 'input', 'raw_input',
            'file', 'execfile', 'reload', '__import__',
            'getattr', 'setattr', 'delattr', 'hasattr',
            'globals', 'locals', 'vars', 'dir'
        }
        
        dangerous_attributes = {
            '__builtins__', '__globals__', '__locals__', '__dict__',
            '__class__', '__bases__', '__subclasses__', '__import__'
        }
        
        # Walk through AST nodes
        for node in ast.walk(tree):
            # Check for dangerous imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in dangerous_imports:
                        return False, f"Dangerous import detected: {alias.name}"
            
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module in dangerous_imports:
                    return False, f"Dangerous import detected: {node.module}"
            
            # Check for dangerous function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in dangerous_functions:
                        return False, f"Dangerous function call detected: {node.func.id}"
            
            # Check for dangerous attribute access
            elif isinstance(node, ast.Attribute):
                if node.attr in dangerous_attributes:
                    return False, f"Dangerous attribute access detected: {node.attr}"
            
            # Check for string-based code execution
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in ['eval', 'exec']:
                    return False, f"Dynamic code execution detected: {node.func.id}"
        
        return True, "Code passed security validation"

    def _generate_analysis_code(self, user_query: str) -> str:
        """Use LLM to understand the query and generate Python analysis code with available libraries."""
        installed_libraries = self._detect_installed_libraries()
        code_generation_prompt = f"""You are a data analyst AI that generates Python code to analyze datasets. 

I have a dataset with {self.dataset_schema['total_records']} records. Here's the dataset structure:

DATASET SCHEMA:
- Fields available: {self.dataset_schema['fields']}
- Field types: {self.dataset_schema['field_types']}
- Categorical fields and their values: {self.dataset_schema['categorical_fields']}

SAMPLE RECORDS:
{json.dumps(self.dataset_schema['sample_records'], indent=2)}

AVAILABLE LIBRARIES:
{', '.join(installed_libraries)}

USER QUERY: "{user_query}"

Generate Python code that will analyze the complete dataset to answer this query. The dataset is available as a variable called `dataset` (a list of dictionaries).

Use the most appropriate libraries from the available ones above for efficient analysis. For example:
- Use pandas for complex data operations and filtering
- Use numpy for numerical computations
- Use collections for counting operations
- Use datetime for date operations

Your code should:
1. Import only the libraries you need from the available list
2. Convert dataset to appropriate format if needed (e.g., pandas DataFrame)
3. Filter/process the dataset as needed
4. Perform the appropriate analysis (counting, grouping, sorting, etc.)
5. Store the results in a variable called `results`
6. Avoid any file I/O or external calls
7. Never use print statements for debugging
8. Never ever hardcode sensitive information (e.g., API keys, passwords)
9. Never ever use commentaries with # on code or annotations

Requirements:
- Use only libraries from the available list above
- Handle edge cases (missing fields, None values, etc.)
- Return results as a dictionary with clear structure
- Include actual data values, not just summaries

Generate ONLY the Python code, no explanations, no comentaries, no documentation:"""
        analysis_code = self.chat_session.send(code_generation_prompt)
        if "```python" in analysis_code: analysis_code = analysis_code.split("```python")[1].split("```")[0]
        elif "```" in analysis_code: analysis_code = analysis_code.split("```")[1].split("```")[0]

        analysis_code = analysis_code.strip()

# SECURITY LAYER 1: Validate code before execution
        is_safe, security_message = self._validate_code_security(analysis_code)
        
        self.logger.debug(f"Generated analysis code: \n{analysis_code}")
        if not is_safe:
            self.logger.warning(f"Security validation failed: {security_message}")
            print(f"Security validation failed: {security_message}")
            raise SecurityError(f"Generated code failed security validation: {security_message}")
        
        self.logger.debug(f"Code passed security validation: {security_message}")
        
        return analysis_code
        # self.logger.debug(f"Generated analysis code: {analysis_code}")
        # return analysis_code

    def _detect_installed_libraries(self) -> List[str]:
        """Detect which data analysis libraries are installed in the environment."""
        potential_libraries = [
            'pandas',
            'numpy', 
            'matplotlib',
            'seaborn',
            'scipy',
            'sklearn',
            'collections',
            'datetime',
            'itertools',
            'operator',
            'functools',
            'statistics',
            're',
            'json',
            'math'
        ]
        
        installed = []
        
        for library in potential_libraries:
            try:
                __import__(library)
                installed.append(library)
            except ImportError:
                continue
        
        self.logger.debug(f"Detected installed libraries: {installed}")
        return installed

    def _execute_analysis_code(self, analysis_code: str) -> Dict[str, Any]:
        """Execute the generated Python analysis code on the complete dataset."""
        try:
            # Prepare the execution environment with detected libraries
            exec_globals = {
                'dataset': self.complete_dataset,
                'results': {},
                'len': len,
                'set': set,
                'list': list,
                'dict': dict,
                'sorted': sorted,
                'sum': sum,
                'max': max,
                'min': min,
            }
            
            exec_globals['__builtins__'] = __builtins__
            
            # Execute the analysis code (it will handle its own imports)
            exec(analysis_code, exec_globals)
            results = exec_globals.get('results', {})

            self.logger.debug(f"Analysis execution completed, results: {type(results)}")
            return results
            
        except Exception as e:
            self.logger.error(f"Code execution failed: {e}")
            self.logger.debug(f"Failed code: {analysis_code}")
            
            return {
                'error': str(e),
                'failed_code': analysis_code,
                'suggestion': 'Code execution failed, may need simpler approach'
            }

    def _format_results_to_response(self, user_query: str, analysis_results: Dict[str, Any]) -> str:
        """Use LLM to format analysis results into natural language response."""
        if 'error' in analysis_results:
            fallback_results = self._fallback_analysis(user_query)
            analysis_results = fallback_results
            self.logger.info("Using fallback analysis due to execution error")

        formatting_prompt = f"""I performed data analysis on a dataset of {len(self.complete_dataset)} records to answer a user's query.

USER QUERY: "{user_query}"

ANALYSIS RESULTS:
{json.dumps(analysis_results, indent=2, default=str)}

Please provide a clear, direct answer to the user's question using these analysis results. 

Requirements:
- Give specific numbers, names, and values from the results
- Be direct and factual
- Don't say "cannot be determined" - use the actual data provided
- Format lists and data clearly
- If there are many results, show the most relevant ones and mention totals

Provide a natural language response:"""
        formatted_response = self.chat_session.send(formatting_prompt)
        return formatted_response

    def _fallback_analysis(self, user_query: str) -> Dict[str, Any]:
        """Perform simple fallback analysis when code generation fails."""
        query_lower = user_query.lower()
        results = {
            'fallback_used': True,
            'total_records': len(self.complete_dataset)
        }
        
        # Look for field names in the query and provide basic stats
        for field in self.dataset_schema['fields']:
            if field.lower() in query_lower:
                values = [record.get(field) for record in self.complete_dataset if record.get(field) is not None]
                if values:
                    unique_values = list(set(values))
                    results[f'{field}_analysis'] = {
                        'total_values': len(values),
                        'unique_values': len(unique_values),
                        'sample_values': unique_values[:10]
                    }
                    if field in self.dataset_schema['categorical_fields']:
                        from collections import Counter
                        counts = Counter(values)
                        results[f'{field}_counts'] = dict(counts.most_common(10))
        return results

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information for overview."""
        return {
            'total_records': len(self.complete_dataset),
            'fields': self.dataset_schema['fields'],
            'sample_record': self.complete_dataset[0] if self.complete_dataset else {}
        }

##########################
# DEFAULT EXECUTION
##########################
global logger, config_file_path
script_dir          = os.path.dirname(os.path.abspath(__file__))
script_name         = os.path.splitext(os.path.basename(sys.argv[0]))[0]
config_file_path    = os.path.join(script_dir, '..', 'conf', f'{script_name}.cfg')
env                 = EnvManager(config_file_path)

# Initialize logger
logging_config      = env.config.get('logging_config', {})
logger              = CustomLogger(config=logging_config, logger_name=script_name)
formatted_config    = "\n".join([f"{key}: {value}" for key, value in env.config.items() if 'API_KEY' not in key])
logger.info("Environment variables and logger initialized successfully")
logger.debug(f"Configuration values set:\n{formatted_config}")

##########################
# CLI INTERFACE
##########################
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python RAGChatSession.py <json_file_path> [context_file_path]")
        print("Example: python RAGChatSession.py ../json_files/cleared_history_data_20250828162315.json")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    context_file_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        chat = GeneralizedRAGChat(env, logger, json_file_path, context_file_path)
        
        info = chat.get_dataset_info()
        print(f"Generalized RAG Dataset Chat Ready")
        print(f"Records loaded: {info['total_records']:,}")
        print(f"Fields available: {info['fields']}")
        print("=" * 80)
        print("Ask me ANYTHING about your dataset - I'll understand and analyze it!")
        print("Type 'info' to see dataset overview, 'exit' to quit.")
        print("=" * 80)
        
        # Main chat loop
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == "exit":
                print("Goodbye!")
                break
            elif user_input.lower() == "info":
                info = chat.get_dataset_info()
                print(f"\nDataset Info:")
                print(f"  Records: {info['total_records']:,}")
                print(f"  Fields: {info['fields']}")
                print(f"  Sample: {info['sample_record']}")
                continue
            elif not user_input:
                continue
            
            try:
                response = chat.send(user_input)
                print(f"\nAssistant: {response}")
            except Exception as e:
                print(f"Error: {e}")
    
    except Exception as e:
        print(f"Failed to initialize: {e}")
        logger.error(f"Initialization error: {e}")
        sys.exit(1)
