from autogen import AssistantAgent, UserProxyAgent
import os
from dotenv import load_dotenv
import autogen



load_dotenv()

llm_config = {"model": "gpt-3.5-turbo", "api_key": os.environ["OPENAI_API_KEY"]}
assistant = AssistantAgent("assistant", llm_config=llm_config)
user_proxy = UserProxyAgent("user_proxy", code_execution_config={"executor": autogen.coding.LocalCommandLineCodeExecutor(work_dir="coding")}, human_input_mode="NEVER", max_consecutive_auto_reply=5, )

# Start the chat
def model_build():
    result = user_proxy.initiate_chat(
        assistant,
        message="""
        you are a python data scientist. you are given tasks to complete that is read the dataset from the working directory with path in working directory uploaded_data.csv  you should do
          model building part chosse the right model and store the model and also store the model_building part code ML_model.py 
    -Build 5 model find top 5 accuracy model and choose one best model atlast also do model evaluation for the data and save the code
    and save all the visualisation in png and also code separately in codebase
    -Save the model_building code in ML_model.py
    - the python code runs in jupyter notebook.
    - every time you call `execute_python` tool, the python code is executed in a separate cell. it's okay to multiple calls to `execute_python`.
    - display visualizations using matplotlib or any other visualization library directly in the notebook. don't worry about saving the visualizations to a file.
    - you have access to the internet and can make api requests.
    - you also have access to the filesystem and can read/write files.
    - you can install any pip package (if it exists) if you need to but the usual packages for data analysis are already preinstalled.
    - you can run any python code you want, everything is running in a secure sandbox environment
        """,
    )

    print(result)


# model_build()

