import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import demo


def test_ask_int_rejects_small_and_bad_values(monkeypatch):
    answers = iter(["5", "abc", "12"])
    monkeypatch.setattr(demo, "read_line", lambda prompt: next(answers))
    assert demo.ask_int("budget", 32, 10) == 12
    monkeypatch.setattr(demo, "read_line", lambda prompt: "")
    assert demo.ask_int("budget", 32, 10) == 32
