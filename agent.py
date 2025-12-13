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
import envs

system_prompt = """
    You are a Smart ReAct agent. You will be solving a small software engineering task in Python.
    The user will provide you with a task description and source code files.

    -- GOALS --
    - You need to analyze the codebase, identify the changes needed to satisfy the issue in the task, and implement those changes by modifying existing files.
    - You should not simply describe a proposed solution, YOU MUST IMPLEMENT IT USING THE TOOLS PROVIDED.

    -- METHODS --
    - At every step, you should REASON about what to do next and then ACT by calling one of your available tools.
    - The available tools are described below, along with their signatures and docstrings.
    - To complete the task, you MUST call the `finish` tool with the final result.

    -- RESPONSES --
    - Every response, including the final result, MUST conform to the response template format provided below.
    - Responses that do not follow the format will be rejected.
    - In the response template below:
        - Lines beginning with "---" are MUST be included exactly as shown in the template.
        - Lines within <angle_brackets> are placeholders that MUST be replaced with your actual thoughts, function names, and argument values.
    
    -- WORKFLOW --
    1. Start with REASONING about the task and the codebase.
    2. Use the available tools to gather information and make changes.
    3. Check your progress by testing your changes.
    4. Repeat the REASONING and ACTING steps as needed.
    5. When you have completed the task, call the `finish` tool with the final result.

    Between every reasoning step, the system will execute your chosen tool and provide you with the output in the message history.

    -- COMPLETION --
    Once you have resolved the user task, call the finish tool with a summary of the changes you made. Before calling finish, ensure that:
        - All code changes are complete and correct.
        - All code changes are made in place in the source files. Simply printing the proposed changes is not sufficient.
        - You have tested your changes and verified that they work as intended.
    If any of these conditions are not met, continue the WORKFLOW until they are.

    -- RULES --
    YOU MUST ALWAYS FOLLOW THESE RULES:
    - Respond using the response template defined in `RESPONSE FORMAT` below.
    - Include your thoughts in every response and a single function call.
    - Only use the tools provided in `AVAILABLE TOOLS` below.
    - Call `finish` ONLY when you have completed the task.
    - DO NOT prompt the user for any clarifications or approvals. Work with the information provided. The user is unable to respond during your reasoning and acting process.
    - Focus on making meaningful progress towards completing the task at all times. A partial solution is better than no solution.
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
                # self.add_message("assistant", response_text)
                self.add_message("assistant", f"The previous response could not be parsed. The correct response format is:\n{self.parser.response_format}\n\nInvalid response:\n{response_text}")
                continue  # Skip to the next iteration if parsing fails
            
            # Add the thought message
            thought_message_id = self.add_message("assistant", parsed_response["thought"])
            
            # Execute function call
            if parsed_response["name"] in self.function_map:
                tool_result = self.function_map[parsed_response["name"]](**parsed_response["arguments"])
                # Add the tool result to the message list
                self.add_message("tool", tool_result)
                if parsed_response["name"] == "finish":
                    # Make sure we actually generated a patch before finishing
                    if "check_patch" in self.function_map:
                        if self.function_map["check_patch"]():
                            return tool_result
                        else:
                            self.add_message("assistant", "The generated patch is empty. I must ensure code changes are complete and correct before finishing.")
                            continue
                    else:
                        return tool_result
            else:
                print(f"Warning: Function {parsed_response['name']} not registered in agent")
                self.add_message("assistant", f"The previous response called an invalid function: '{parsed_response['name']}'. The available tools are described in the system prompt.\n\nInvalid response:\n{response_text}")
                continue

        self.finish("Reached max_steps without calling finish")

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
