"""Runs the analyst agent on questions with known answers and scores it. Re-run after every prompt change."""
import re

import agent

CASES = [
    ("How many events are there?", "2683"),
    ("How many events had 500,000 or more customer-hours?", "649"),
    ("How many events were there in 2020?", "454"),
    ("Which month has the most events?", "february"),
    ("Which state lost the most customer-hours?", "florida"),
]

def normalise(text):
    return re.sub(r"(?<=\d),(?=\d)", "", text.lower())   # "2,683" becomes "2683"

auto = lambda name, args: agent.run_python(args["code"], confirm=False)
passed = 0
for question, expected in CASES:
    ok = expected in normalise(agent.ask(question, run_tool=auto))
    passed += ok
    print("PASS" if ok else "FAIL", question)
print(f"{passed}/{len(CASES)} passed")