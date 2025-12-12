"""
Starter scaffold for the CS 294-264 HW1 ReAct agent.

Students must implement a minimal ReAct agent that:
- Maintains a message history list (role, content, timestamp, unique_id)
- Uses a textual function-call format (see ResponseParser) with rfind-based parsing
- Alternates Reasoning and Acting until calling the tool `finish`
- Supports tools: `run_bash_cmd`, `finish`

This file intentionally omits core implementations and replaces them with
clear specifications and TODOs.
"""

from typing import List, Callable, Dict, Any
import time

from response_parser import ResponseParser
from llm import LLM, OpenAIModel
from envs import LimitsExceeded
import inspect

system_prompt = """
    You are a Smart ReAct agent. You will be solving a small software engineering task in Python.
    At every step, you will REASON about what to do next, and then you will ACT by calling one of your available TOOLS.
    The user will provide you with a task description and source code files.
    At every step, you MUST think about what to do next and then act by calling one of your available tools.
    Use the available tools to gather information about the codebase and modify files as needed.
    The tools you can use are described below, along with their signatures and docstrings.
    When you have completed the task, you MUST call the `finish` tool with the final result.
    Every response, including the final result, MUST conform to the response format provided below.

    To complete the user task, follow these steps:

    INITIAL SETUP:
    Begin by performing the following steps once:
    1. Carefully read the user task and identify the requirements.
    2. Read and understand the relevant parts of the codebase:
        - Use `run_bash_cmd` to identify files (e.g. `ls -la`)
        - Use `show_file` to read file contents
    3. Analyze the code and identify what changes are needed to fulfill the task.

    MAIN LOOP:
    In each iteration of the loop, you should do either step 4 or step 5:
    4. Iteratively make the necessary code changes by writing new code or modifying existing code.
        - Use `replace_in_file` to modify existing files
        - Use `create_file` to create new files
    5. Test your changes by running relevant tests or writing new tests.
        - Use `run_bash_cmd` to run tests (e.g. `pytest tests/`, `python -m unittest discover`, `python test_script.py`)
        - If tests fail or issues arise, REASON about the cause and ACT to fix them.

    COMPLETION:
    6. Once you have completed the user task, call the finish tool with a summary of the changes you made. Before calling finish, ensure that:
        - All code changes are complete and correct.
        - All code changes are made in place in the source files. Simply printing the proposed changes is not sufficient.
        - You have tested your changes and verified that they work as intended.
    If any of these conditions are not met, continue the MAIN LOOP until they are.

    Always follow these rules:
    - You MUST ALWAYS respond using the specified response format.
    - You MUST call one tool at a time.
    - You MUST NOT make up any tools; only use the ones provided.
    - You MUST call the `finish` tool when you have completed the task.
    - You MUST ENSURE that your final code changes are syntactically correct and follow Python conventions. Python is highly sensitive to indentation and syntax errors.
    - DO NOT prompt the user for any clarifications or approvals. Work with the information provided. The user is unable to respond during your reasoning and acting process.

    It is crucial that you adhere strictly to the response format and tool usage guidelines provided.

    Note: It is possible that you will not be able to complete the task within the given step limit. In such cases, you should call the `finish` tool with the best result you have achieved so far.
    A partial solution is better than no solution.
"""

class ReactAgent:
    """
    Minimal ReAct agent that:
    - Maintains a message history list with unique ids
    - Builds the LLM context from the message list
    - Registers callable tools with auto-generated docstrings in the system prompt
    - Runs a Reason-Act loop until `finish` is called or MAX_STEPS is reached
    """

    def __init__(self, name: str, parser: ResponseParser, llm: LLM):
        self.name: str = name
        self.parser = parser
        self.llm = llm

        # Message list storage
        self.id_to_message: List[Dict[str, Any]] = []
        self.root_message_id: int = -1
        self.current_message_id: int = -1

        # Registered tools
        self.function_map: Dict[str, Callable] = {}

        # Set up the initial structure of the history
        # Create required root nodes and a user node (task)
        self.system_message_id = self.add_message("system", system_prompt)
        self.user_message_id = self.add_message("user", "")
        # NOTE: mandatory finish function that terminates the agent
        self.add_functions([self.finish])

    # -------------------- MESSAGE LIST --------------------
    def add_message(self, role: str, content: str) -> int:
        """
        Create a new message and add it to the list.

        The message must include fields: role, content, timestamp, unique_id.
        """
        # Increment the unique message ID
        self.current_message_id += 1
        # Push the new message to the list
        self.id_to_message.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "unique_id": self.current_message_id,  # Unique ID
        })
        # Verify the message ID is consistent
        if self.current_message_id != len(self.id_to_message) - 1:
            print(f"Warning: current_message_id ({self.current_message_id}) does not match the last index in id_to_message ({len(self.id_to_message) - 1})")
            # raise ValueError("Message ID mismatch: current_message_id must match the last index in id_to_message")
        return self.current_message_id

    def set_message_content(self, message_id: int, content: str) -> None:
        """
        Update message content by id.
        """
        if message_id < 0 or message_id >= len(self.id_to_message):
            print(f"Warning: Attempted to set content for invalid message ID {message_id}")
            return
            # raise IndexError(f"Message ID {message_id} out of range")
        self.id_to_message[message_id]["content"] = content

    def get_context(self) -> str:
        """
        Build the full LLM context from the message list.
        The context includes all messages except the system and initial user task message.
        """
        context = ""
        for id in range(self.current_message_id + 1):
            if id == self.system_message_id or id == self.user_message_id:
                continue  # Skip system and user messages in context
            msg_context = self.message_id_to_context(id)
            context += msg_context
        return context

    # -------------------- REQUIRED TOOLS --------------------
    def add_functions(self, tools: List[Callable]):
        """
        Add callable tools to the agent's function map.

        The system prompt must include tool descriptions that cover:
        - The signature of each tool
        - The docstring of each tool
        """
        # NOTE: The tool descriptions are generated automatically and added to the system prompt in message_id_to_context.
        for tool in tools:
            self.function_map[tool.__name__] = tool
    
    def finish(self, result: str):
        """The agent must call this function with the final result when it has solved the given task. The function calls "git add -A and git diff --cached" to generate a patch and returns the patch as submission.

        Args: 
            result (str); the result generated by the agent

        Returns:
            The result passed as an argument.  The result is then returned by the agent's run method.
        """
        return result 

    # -------------------- MAIN LOOP --------------------
    def run(self, task: str, max_steps: int) -> str:
        """
        Run the agent's main ReAct loop:
        - Set the user prompt
        - Loop up to max_steps (<= 100):
            - Build context from the message list (with `message_id_to_context`)
            - Query the LLM
            - Parse a single function call at the end (see ResponseParser)
            - Execute the tool
            - Append tool result to the list
            - If `finish` is called, return the final result
        """
        # Set the user task message
        self.set_message_content(self.user_message_id, task)
        
        # Safety check for max_steps
        if max_steps > 100:
            print("Warning: max_steps should not exceed 100 for ReAct agents, defaulting to 100")
            max_steps = 100
            # raise ValueError("max_steps must be <= 100")
        
        # Run the ReAct loop
        for _ in range(max_steps):
            system_prompt = self.message_id_to_context(self.system_message_id)
            user_prompt = self.message_id_to_context(self.user_message_id)
            # Build the context from the message list
            context = self.get_context()
            # Query the LLM with the current context
            response_text = self.llm.generate([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}, {"role": "assistant", "content": context}])
            
            # Parse the response to extract the function call
            try:
                parsed_response = self.parser.parse(response_text)
            except ValueError as e:
                print(f"Error parsing response: {e}")
                print(response_text)
                self.add_message("user", f"The previous response could not be parsed. Please use the correct response format:\n{self.parser.response_format}")
                continue  # Skip to the next iteration if parsing fails
            
            # Add the thought message
            thought_message_id = self.add_message("assistant", parsed_response["thought"])
            
            # Execute function call
            if parsed_response["name"] in self.function_map:
                tool_result = self.function_map[parsed_response["name"]](**parsed_response["arguments"])
                # Add the tool result to the message list
                self.add_message("tool", tool_result)
                if parsed_response["name"] == "finish":
                    # If finish was called, return the result
                    return tool_result
            else:
                print(f"Warning: Function {parsed_response['name']} not registered in agent")
                self.add_message("user", f"The function '{parsed_response['name']}' is not recognized. Please use one of the available tools.")
                continue
                # raise ValueError(f"Function {parsed_response['name']} not registered in agent")

        self.finish("Reached max_steps without calling finish")
        # raise LimitsExceeded(f"Reached max_steps ({max_steps}) without calling finish")

    def message_id_to_context(self, message_id: int) -> str:
        """
        Helper function to convert a message id to a context string.
        """
        message = self.id_to_message[message_id]
        header = f'----------------------------\n|MESSAGE(role="{message["role"]}", id={message["unique_id"]})|\n'
        content = message["content"]
        if message["role"] == "system":
            tool_descriptions = []
            for tool in self.function_map.values():
                signature = inspect.signature(tool)
                docstring = inspect.getdoc(tool)
                tool_description = f"Function: {tool.__name__}{signature}\n{docstring}\n"
                tool_descriptions.append(tool_description)

            tool_descriptions = "\n".join(tool_descriptions)
            return (
                f"{header}{content}\n"
                f"--- AVAILABLE TOOLS ---\n{tool_descriptions}\n\n"
                f"--- RESPONSE FORMAT ---\n{self.parser.response_format}\n"
            )
        else:
            return f"{header}{content}\n"

def main():
    from envs import DumbEnvironment
    llm = OpenAIModel("----END_FUNCTION_CALL----", "gpt-4o-mini")
    parser = ResponseParser()

    env = DumbEnvironment()
    dumb_agent = ReactAgent("dumb-agent", parser, llm)
    dumb_agent.add_functions([env.run_bash_cmd])
    result = dumb_agent.run("Show the contents of all files in the current directory.", max_steps=10)
    # result = dumb_agent.run("List the names of all files in the current directory.", max_steps=10)
    print(result)

if __name__ == "__main__":
    # Optional: students can add their own quick manual test here.
    main()
