from agent import run_python

def test_returns_output():
    assert run_python("print(1 + 1)", confirm=False).strip() == "2"

def test_reports_errors_instead_of_crashing():
    assert "ZeroDivisionError" in run_python("1 / 0", confirm=False)

def test_times_out():
    assert "longer than 30 seconds" in run_python("import time; time.sleep(31)", confirm=False)