from utils import get_sb_environment
import subprocess
import os
import swebench

class LimitsExceeded(Exception):
    """Raised when the agent has reached its step limit."""


class SWEEnvironment:
    """
    Minimal interface to the SWEBench execution environment.

    Students may use their own wrapper. The environment must expose:
    - execute(command: str) -> str: Run a shell command and return stdout, or raise ValueError on failure
    """

    def __init__(self, instance: dict):
        self.env = get_sb_environment(instance)
        self.instance = instance  # Store instance for test execution
     
    # -------------------- REQUIRED TOOLS --------------------
    def run_bash_cmd(self, command: str) -> str:
        """
        Run the command in a bash shell and return the output. If the command fails (non-zero exit code),
        catch the exception and return the partial output.

        Args;
            command (str): the shell command to run

        Returns:
            The output of running the shell command
        """
        try:
            output = self.env.execute(command)
            
            # Handle case where execute returns a dict instead of string
            if isinstance(output, dict):
                output = output.get("output", "") or output.get("stdout", "")
                
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            return f"Command timed out with: {e}.\n\nPartial output:\n{output}"
        except TimeoutError:
            return f"Command time out.\n\nPartial output:\n{output}"
        return output
    
    def generate_patch(self, result: str) -> str:
        """
        Generate a patch from the result (for SWE-Bench)
        """
        try:
            patch_output = self.env.execute("git add -A && git diff --cached")
            
            # Handle case where execute returns a dict instead of string
            if isinstance(patch_output, dict):
                patch_output = patch_output.get("output", "") or patch_output.get("stdout", "")
            
            if patch_output and patch_output.strip():
                return patch_output
            else:
                return f"{result}\n\nNo changes detected to generate a patch."
        except Exception as e:
            return f"{result}\n\nError running git commands: {e}"

    def check_patch(self) -> bool:
        """
        Check for a non-empty patch from the current changes. Used to avoid submitting empty patches.

        Args:
            result (str): The result string (not used here but kept for consistency)
        Returns:
            True if there are changes to be committed, False otherwise.
        """
        try:
            patch_output = self.env.execute("git add -A && git diff --cached")
            
            # Handle case where execute returns a dict instead of string
            if isinstance(patch_output, dict):
                patch_output = patch_output.get("output", "") or patch_output.get("stdout", "")
            
            if patch_output and patch_output.strip():
                return True
            else:
                return False
        except Exception as e:
            return False

    # -------------------- TODO(student): add more functions here if you want, not required --------------------
    def replace_in_file(self, file_path: str, from_line: int, to_line: int, content: str) -> str:
        """
        Replace the content of the file from lines from_line to to_line with the given content.

        Notes:
            - Line numbers are 1-indexed (inclusive).
            - Validates bounds and preserves the rest of the file.
            - Ensures final file ends with a newline where appropriate.

        Args:
            file_path (str): path to the file
            from_line (int): starting line number (1-indexed)
            to_line (int): ending line number (1-indexed)
            content (str): content to replace with

        Returns:
            The updated content of the file after replacement or a failure message.
        """
        if not os.path.isfile(file_path):
            return f"File not found: {file_path}"
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            # Adjust for 0-indexing
            from_idx = max(0, from_line - 1)
            to_idx = min(len(lines), to_line)
            if from_idx > len(lines) or from_idx < 0:
                return f"Invalid from_line: {from_line} for file with {len(lines)} lines"
            if to_idx < from_idx:
                return f"Invalid range: to_line ({to_line}) must be >= from_line ({from_line})"
            # Replace the specified lines
            # Normalize content newline
            replacement = content if content.endswith('\n') else content + '\n'
            new_lines = lines[:from_idx] + [replacement] + lines[to_idx:]
            with open(file_path, 'w') as f:
                f.writelines(new_lines)
            return ''.join(new_lines)
        except Exception as e:
            return f"Error replacing content in file {file_path}: {e}"

    def show_file(self, file_path: str) -> str:
        """
        Show the content of the file.

        Args:
            file_path (str): path to the file

        Returns:
            The content of the file.
        """
        if not os.path.isfile(file_path):
            return f"File not found: {file_path}"
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            return content
        except Exception as e:
            return f"Error reading file {file_path}: {e}"

    def create_file(self, file_path: str, content: str) -> str:
        """
        Create a new file with the given content.

        Args:
            file_path (str): path to the file
            content (str): content to write to the file

        Returns:
            A message indicating success or failure.
        """
        try:
            with open(file_path, 'w') as f:
                f.write(content)
            return f"File created: {file_path}"
        except Exception as e:
            return f"Error creating file {file_path}: {e}"

    def append_to_file(self, file_path: str, content: str) -> str:
        """
        Append content to the end of a file, creating it if missing.

        Args:
            file_path (str): path to the file
            content (str): content to append

        Returns:
            The updated tail of the file or an error message.
        """
        try:
            os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
            with open(file_path, 'a+') as f:
                f.write(content)
            with open(file_path, 'r') as f:
                tail = f.read()[-5000:]
            return tail
        except Exception as e:
            return f"Error appending to file {file_path}: {e}"

    def list_dir(self, path: str) -> str:
        """
        List files and directories in a given directory.

        Note: non-recursive, only returns immediate children.

        Args:
            path (str): directory to list. Defaults to current directory.

        Returns:
            A newline-delimited list of files and directories.
        """
        try:
            return '\n'.join(os.listdir(path))
        except Exception as e:
            return f"Error listing {path}: {e}"

    def list_python_files(self, path: str = '.') -> str:
        """
        List all Python files in a given directory.

        Args:
            path (str): directory to list. Defaults to current directory.

        Returns:
            A newline-delimited list of Python files.
        """
        try:
            return '\n'.join(f for f in os.listdir(path) if f.endswith('.py'))
        except Exception as e:
            return f"Error listing Python files in {path}: {e}"

    def grep_in_file(self, file_path: str, pattern: str) -> str:
        """
        Return lines in a file that match a regex pattern (Python re).

        Args:
            file_path (str): path to the file
            pattern (str): Python regex pattern

        Returns:
            Matching lines with line numbers, or an error message.
        """
        import re
        if not os.path.isfile(file_path):
            return f"File not found: {file_path}"
        try:
            regex = re.compile(pattern)
            matches = []
            with open(file_path, 'r') as f:
                for i, line in enumerate(f, start=1):
                    if regex.search(line):
                        matches.append(f"{i}: {line.rstrip()}")
            return '\n'.join(matches) if matches else ""
        except Exception as e:
            return f"Error grepping file {file_path}: {e}"

    def check_python_syntax(self, file_path: str) -> str:
        """
        Check Python syntax by attempting to compile the file.

        Returns an empty string if valid; otherwise returns the error message.
        """
        if not os.path.isfile(file_path):
            return f"File not found: {file_path}"
        try:
            with open(file_path, 'r') as f:
                source = f.read()
            compile(source, file_path, 'exec')
            return ""
        except Exception as e:
            return f"Syntax error in {file_path}: {e}"

class DumbEnvironment:
    """
    Dumb environment that just executes the command
    """

    def execute(self, command: str) -> str:
        """
        Run the command in bash and return the output

        Args;
            command (str): the shell command to run

        Returns:
            The output of running the shell command
        """
        result = subprocess.run(command, capture_output=True, shell=True, check=False)
        output = f"--STDOUT--\n{result.stdout.decode()}\n--STDERR--\n{result.stderr.decode()}"
        if result.returncode:
            raise ValueError(output)
        return output
    
    def run_bash_cmd(self, command: str) -> str:
        """
        Run the command in a bash shell and return the output or throw a ValueError
        if the process returns non-zero exit code.

        Args;
            command (str): the shell command to run

        Returns:
            The output of running the shell command
        """
        return self.execute(command)
