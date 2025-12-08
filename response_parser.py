class ResponseParser:
    """
    Parses LLM responses to extract a single function call using a rigid textual format.

    The LLM must output exactly one function call at the end of its response.
    Do NOT use JSON or XML. Use rfind to locate the final markers.
    """

    BEGIN_CALL = "----BEGIN_FUNCTION_CALL----"
    END_CALL = "----END_FUNCTION_CALL----"
    ARG_SEP = "----ARG----"
    VALUE_SEP = "----VALUE----"

    # Students should include this exact template in the system prompt so the LLM follows it.
    response_format = f"""
your_thoughts_here
...
{BEGIN_CALL}
function_name
{ARG_SEP}
arg1_name
{VALUE_SEP}
arg1_value (can be multiline)
{ARG_SEP}
arg2_name
{VALUE_SEP}
arg2_value (can be multiline)
...
{END_CALL}

DO NOT CHANGE ANY TEST! AS THEY WILL BE USED FOR EVALUATION.
"""

    def parse(self, text: str) -> dict:
        """
        Parse the function call from `text` using string.rfind to avoid confusion with
        earlier delimiter-like content in the reasoning.

        Returns a dictionary: {"thought": str, "name": str, "arguments": dict}
        
        The parser strictly follows the required textual protocol and only extracts the
        last function call block delimited by BEGIN/END markers.
        """
        if not isinstance(text, str):
            raise ValueError("Input to ResponseParser.parse must be a string")

        # Find the last END marker; ensure it exists
        end_idx = text.rfind(self.END_CALL)
        if end_idx == -1:
            raise ValueError("END function call marker not found")

        # Find the last BEGIN marker that precedes the END marker
        begin_idx = text.rfind(self.BEGIN_CALL, 0, end_idx)
        if begin_idx == -1:
            raise ValueError("BEGIN function call marker not found")

        # Thought is everything before the BEGIN marker
        thought = text[:begin_idx].rstrip()

        # Implement backward parsing using rfind per the specified logic
        begin_content_idx = begin_idx + len(self.BEGIN_CALL)
        arguments: dict[str, str] = {}

        # Iterate backwards over ARG/VALUE blocks
        while True:
            arg_idx = text.rfind(self.ARG_SEP, begin_content_idx, end_idx)
            if arg_idx == -1:
                break
            val_idx = text.rfind(self.VALUE_SEP, arg_idx + len(self.ARG_SEP), end_idx)
            if val_idx == -1 or val_idx <= arg_idx:
                # Malformed block; stop parsing further
                raise ValueError("Malformed function call block; missing ARG or VALUE separator")

            # Extract arg name and value slices
            arg_name_section = text[arg_idx + len(self.ARG_SEP):val_idx]
            value_section = text[val_idx + len(self.VALUE_SEP):end_idx]

            # Normalize by stripping leading/trailing whitespace; preserve internal content
            arg_name = arg_name_section.strip()
            value = value_section.strip()
            arguments[arg_name] = value

            # Move the window backward to before this ARG block
            end_idx = arg_idx

        # After consuming all ARG blocks, the function name is the first non-empty line
        # between BEGIN content index and the current end_idx
        name_window = text[begin_content_idx:end_idx]
        # Split and find first non-empty line
        name = None
        for line in name_window.splitlines():
            if line.strip() != "":
                name = line.strip()
                break
        if not name:
            raise ValueError("Function name not found in function call block")

        return {"thought": thought, "name": name, "arguments": arguments}

def test_basic_two_args():
    parser = ResponseParser()
    text = (
        "Reasoning about what to do...\n"
        + ResponseParser.BEGIN_CALL + "\n"
        + "run_bash_cmd\n"
        + ResponseParser.ARG_SEP + "\n"
        + "command\n"
        + ResponseParser.VALUE_SEP + "\n"
        + "echo hello\n"
        + ResponseParser.ARG_SEP + "\n"
        + "extra\n"
        + ResponseParser.VALUE_SEP + "\n"
        + "line1\nline2\n"
        + ResponseParser.END_CALL
    )
    out = parser.parse(text)
    assert out["name"] == "run_bash_cmd"
    assert out["arguments"]["command"] == "echo hello"
    assert out["arguments"]["extra"] == "line1\nline2"
    assert "Reasoning" in out["thought"]

def test_no_args():
    parser = ResponseParser()
    text = (
        "My plan\n"
        + ResponseParser.BEGIN_CALL + "\n"
        + "finish\n"
        + ResponseParser.END_CALL
    )
    out = parser.parse(text)
    assert out["name"] == "finish"
    assert out["arguments"] == {}

def test_multiple_calls_parse_last():
    parser = ResponseParser()
    first = (
        ResponseParser.BEGIN_CALL + "\n"
        + "finish\n"
        + ResponseParser.END_CALL
    )
    second = (
        ResponseParser.BEGIN_CALL + "\n"
        + "run_bash_cmd\n"
        + ResponseParser.ARG_SEP + "\n"
        + "command\n"
        + ResponseParser.VALUE_SEP + "\n"
        + "pwd\n"
        + ResponseParser.END_CALL
    )
    text = "Thoughts\n" + first + "\nmore thoughts\n" + second
    out = parser.parse(text)
    assert out["name"] == "run_bash_cmd"
    assert out["arguments"]["command"] == "pwd"

def test_whitespace_handling():
    parser = ResponseParser()
    text = (
        "Pre\n\n" + ResponseParser.BEGIN_CALL + "\n\n"
        + "  run_bash_cmd  \n\n"
        + ResponseParser.ARG_SEP + "\n\n"
        + "  command  \n\n"
        + ResponseParser.VALUE_SEP + "\n\n"
        + "  ls -la\n\n"
        + ResponseParser.END_CALL
    )
    out = parser.parse(text)
    assert out["name"] == "run_bash_cmd"
    assert out["arguments"]["command"] == "ls -la"

def test_error_missing_end():
    parser = ResponseParser()
    text = ResponseParser.BEGIN_CALL + "\nfinish\n"  # missing END
    try:
        parser.parse(text)
        assert False, "Expected ValueError for missing END"
    except ValueError as e:
        assert "END" in str(e)

def test_error_malformed_block():
    parser = ResponseParser()
    # Missing VALUE_SEP after ARG_SEP
    text = (
        ResponseParser.BEGIN_CALL + "\n"
        + "run_bash_cmd\n"
        + ResponseParser.ARG_SEP + "\n"
        + "command\n"
        + ResponseParser.END_CALL
    )
    try:
        parser.parse(text)
        assert False, "Expected ValueError for malformed block"
    except ValueError as e:
        assert "Malformed" in str(e)

if __name__ == "__main__":
    tests = [
        test_basic_two_args,
        test_no_args,
        test_multiple_calls_parse_last,
        test_whitespace_handling,
        test_error_missing_end,
        test_error_malformed_block,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"Test {t.__name__} passed")
            passed += 1
        except AssertionError as e:
            print(f"Test {t.__name__} failed: {e}")
            continue  # Skip to next test on failure
    print(f"ResponseParser tests passed: {passed}/{len(tests)}")