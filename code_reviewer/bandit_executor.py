import os
import subprocess

BASE_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")


def bandit_process(path: str) -> str:
    # Define the path to the directory you want to analyze
    directory_path = path

    # Define the output file path
    output_file = os.path.join(BASE_PATH, "data/bandit/bandit_report.txt")

    # Construct the command as a list of strings
    command = [
        "bandit",
        "-r",  # Recursive analysis
        directory_path,
        "-f",
        "txt",  # Output format set as text
        "-o",
        output_file,  # Output file
    ]

    # Run the command
    result = subprocess.run(command, text=True, capture_output=True)

    # Check if there was an error
    if result.returncode != 0:
        print("Error:", result.stderr)
    else:
        print("Output saved to", output_file)

    return output_file


if __name__ == "__main__":
    bandit_process(
        "/Users/ricardo/DEV/secure-code-review-demo-app/data/code_to_review/"
    )
