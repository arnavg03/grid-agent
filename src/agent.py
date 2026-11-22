"""Ask the grid: an agent that answers questions about the outage data by writing and running pandas code"""

import json
import subprocess
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()                                       # reads .env file in the repo root
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5-5"
ROOT = Path(__file__).resolve().parent.parent       # repo root, so it runs from any folder
WORK = ROOT / "work"                               # data folder; charts saved here

SYSTEM = """You analyse US power outage data (DOE-417 major events, Nov 2014 to 2023). events.csv has one row per event: EventKey, Year, State, EventType, 
Began, CountyOutages, CustomerHours, Weather (Yes/No), Month, HighImpact (CustomerHours >= 500,000). Answer by calling run_python. Print every number you
rely. Save any chart as a PNG. If the data can't answer the question, say so. End with the answer, the key numbers and one caveat.
outages.csv has one row per distinct county outage: fips, state, county, start_time, duration, mean_customers, CustomerHours. Overlapping events share   
county outages, so never add CustomerHours across events for a total or a ranking; use outages.csv instead."""  # this line is special because it added more context that i noticed ab the data.

TOOLS = [{
    "name": "run_python",
    "description": "Runs a complete Python script (pandas and matplotlib installed) in the folder that holds the data files. Returns everything the script prints, or the error message.",
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "A complete python script"
            }
        },
        "required": ["code"]
    }
}]

def run_python(code, confirm =True):
    if confirm:
        print("\n --- the model wants to run --- \n", code)
        if input("Run it? (y/n) ").strip().lower() != "y":
            return "the user declined to run the code."
    (WORK / "step.py").write_text(code)
    try:
        out = subprocess.run([sys.executable, "-I", "step.py"], cwd=WORK, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return "Error: the script took longer than 30 seconds."
    return (out.stdout + out.stderr)[-4000:]

def log(entry):
    with open(ROOT / "agent_log.jsonl", "a") as f:
        f.write(json.dumps(entry, default = str) + "\n")

def ask(question, system = SYSTEM, tools = TOOLS, run_tool = None, max_steps = 8):
    run_tool = run_tool or (lambda name, args: run_python(args["code"]))
    messages = [{"role": "user", "content": question}]
    start = time.time()
    for step in range(max_steps):
        response = client.messages.create(model= MODEL, max_tokens = 2000, system = system, tools = tools, messages = messages)
        messages.append({"role": "assistant", "content": response.content})
        log({"question": question, "step": step, "tokens_in" : response.usage.input_tokens, "tokens_out": response.usage.output_tokens})
        if response.stop_reason != "tool_use":
            answer = "".join(b.text for b in response.content if b.type == "text")
            log({"question": question, "answer": answer, "seconds": round(time.time() - start, 1)})
            return answer
        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = run_tool(block.name, block.input)
                log({"question": question, "tool": block.name, "input": block.input, "output": output[:500]})
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})
    return "Stoppped: step limit reached."

if __name__ == "__main__":
    print(ask(" ".join(sys.argv[1:]) or "How many events had 500,000 or more customer-hours?"))