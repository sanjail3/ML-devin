import json
import logging

from hashlib import md5
from typing import Optional
from e2b import Sandbox
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from dotenv import load_dotenv

load_dotenv()


sandbox = Sandbox(template="base")

logger = logging.getLogger(__name__)


def execute_code(
    code: str,
    sandbox: Sandbox,
    timeout: Optional[int] = None,
    work_dir: Optional[str] = "coding",
    packages: Optional[str] = None,
):
    if packages is not None and packages != "":
        sandbox.process.start_and_wait(f"pip install -qq {packages}")

    code_hash = md5(code.encode()).hexdigest()
    filename = f"{work_dir}/{code_hash}.py"
    sandbox.filesystem.write(filename, code)

    proc = sandbox.process.start_and_wait(
        f"python {filename}",
        timeout=timeout,
        cwd=work_dir,
    )

    if proc.exit_code > 0:
        raise Exception(proc.stderr)
    return proc.stdout

config_list = config_list_from_json(
   "OAI_CONFIG_LIST.txt",
    filter_dict={

        "model": ["gpt-3.5-turbo"],


    }
)

llm_config = {
    "functions": [
        {
            "name": "define_function",
            "description": "Define a function to add to the context of the conversation. Necessary Python packages must be declared. Once defined, the assistant may decide to use this function, respond with a normal message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the function to define.",
                    },
                    "description": {
                        "type": "string",
                        "description": "A short description of the function.",
                    },
                    "arguments": {
                        "type": "string",
                        "description": "JSON schema of arguments encoded as a string. For example: { \"url\": { \"type\": \"string\", \"description\": \"The URL\", }}",
                    },
                    "packages": {
                        "type": "string",
                        "description": "A list of space separated package names imported by the function, and that need to be installed with pip prior to invoking the function, for example `requests`. This solves ModuleNotFoundError.",
                    },
                    "code": {
                        "type": "string",
                        "description": "The implementation in Python. Do not include the function declaration.",
                    },
                },
                "required": ["name", "description", "arguments", "packages", "code"],
            },
        },
    ],
    "config_list": config_list,
}


def define_function(name, description, arguments, packages, code):
    json_args = json.loads(arguments)
    function_config = {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": json_args},
        "required": list(json_args.keys()),
    }
    llm_config["functions"] = llm_config["functions"] + [function_config]
    user_proxy.register_function(function_map={name: lambda **args: execute_func(name, packages, code, **args)})
    assistant.update_function_signature(function_config, is_remove=False)
    return f"A function has been added to the context of this conversation.\nDescription: {description}"


def execute_func(name, packages, code, **args):
    str = f"""
print("Result of {name} function execution:")
{code}
args={args}
result={name}(**args)
if result is not None: print(result)
"""
    print(f"execute_code:\n{str}")
    result = execute_code(str, sandbox=sandbox, timeout=120, packages=packages)
    print(f"Result: {result}")
    return result


def _is_termination_msg(message):
    if isinstance(message, dict):
        message = message.get("content")
        if message is None:
            return False
        return message.rstrip().endswith("TERMINATE")


assistant = AssistantAgent(
    name="chatbot",
    system_message="""You are an assistant.
        The user will ask a question.
        You may use the provided functions before providing a final answer.
        Only use the functions you were provided.
        When the answer has been provided, reply TERMINATE.""",
    llm_config=llm_config,
)

user_proxy = UserProxyAgent(
    "user_proxy",
    code_execution_config=False,
    is_termination_msg=_is_termination_msg,
    default_auto_reply="Reply TERMINATE when the initial request has been fulfilled.",
    human_input_mode="NEVER",
llm_config=llm_config
)


user_proxy.register_function(
    function_map={
        "define_function": define_function
    }
)

def main():
    while True:
        message = input("What task would you like executed?\n\n> ")
        print("\n")
        if message in ["exit", "TERMINATE"]:
            print("Exiting...")
            sandbox.close()
            break

        user_proxy.initiate_chat(
            assistant, message=message,)



# main()