# Author : Gabriel Santini Francisco
# Email  : gabrielsantinifrancisco@outlook.com

# Description:
#     This script defines the OpenAIChatSession class, which manages conversational sessions with the OpenAI Chat API.
#     It handles environment configuration, logging, context initialization from a file, message history management,
#     and sending user messages to the OpenAI API. The script also supports interactive command-line usage for chatting.


import os, requests, sys, traceback
from CustomLogger import CustomLogger
from EnvManager import EnvManager

class OpenAIChatSession:
    """
    OpenAIChatSession manages a conversational session with the OpenAI Chat API, maintaining message history and handling context initialization.

    Attributes:
        env (EnvManager): Environment manager providing configuration values.
        logger (CustomLogger): Logger for recording events and errors.
        model (str): The OpenAI model to use for chat completion.
        api_key (str): API key for authenticating with the OpenAI API.
        url (str): Endpoint URL for the OpenAI Chat API.
        cert_path (str): Path to the SSL certificate for secure requests.
        messages (list): List of message dictionaries representing the conversation history.

    Methods:
        __init__(env, logger, context_file_path=None):
            Initializes the chat session, loads context from a file if provided, and sets up configuration.

        send(user_message):
            Sends a user message to the OpenAI API, updates the conversation history, and returns the assistant's reply.
    """
    def __init__(self, env: EnvManager, logger: CustomLogger, context_file_path: str = None) -> None:
        """Initializes the class with environment configuration and logger, setting up model, API key, URL, certificate path, and message storage."""
        self.env = env
        self.logger = logger
        self.model = env.config['OPENAI_MODEL']
        self.api_key = env.config['OPENAI_API_KEY']
        self.url = env.config['CHAT_URL']
        self.cert_path = env.config['cert_path']
        self.messages = []

        if context_file_path:
            try:
                with open(context_file_path, 'r') as file: context = file.read().strip()
                if not context:
                    self.logger.warning(f"Context file {context_file_path} is empty.")
                    print(f"Context file {context_file_path} is empty.")
                    return
                self.messages.append({"role": "system", "content": context})
                self.logger.info(f"Loaded context from {context_file_path}")
                self.logger.debug(f"Context content: {context}")
            except Exception as e:
                self.logger.error(f"Failed to read context file {context_file_path}: {e}")
                self.logger.debug(f'\n{traceback.format_exc()}')
                print(traceback.format_exc())
                return

    def send(self, user_message: str) -> str:
        """Sends a user message to the OpenAI API, updates conversation history, and returns the assistant's reply."""
        self.messages.append({"role": "user", "content": user_message})
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": self.messages
        }
        self.logger.info(f"Sending message to OpenAI: {user_message}")
        self.logger.debug(f"Current message history: {self.messages}")
        response = requests.post(self.url, headers=headers, json=data, verify=self.cert_path)
        try:
            response.raise_for_status()
            self.logger.debug(f"Response: {response.json()}")
            reply = response.json()["choices"][0]["message"]["content"]
            self.messages.append({"role": "assistant", "content": reply})
            return reply
        except requests.exceptions.JSONDecodeError:
            self.logger.error(f"Non-JSON response: {response.text}")
            self.logger.debug(f'\n{traceback.format_exc()}')
            raise
        except Exception as e:
            self.logger.error(f"Request failed: {e}, Response: {response.text}")
            self.logger.debug(f'\n{traceback.format_exc()}')
            raise

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

if __name__ == "__main__":
    context_path = sys.argv[1] if len(sys.argv) > 1 else None
    chat = OpenAIChatSession(env, logger, context_file_path=context_path)
    print("Type 'exit' to end the conversation.")
    while True:
        user_prompt = input("You: ")
        if user_prompt.strip().lower() == "exit": break
        reply = chat.send(user_prompt)
        print("ChatGPT:", reply)
