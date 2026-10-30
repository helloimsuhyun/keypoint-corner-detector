from __future__ import annotations

import importlib


def test_package_main_forwards_arguments_and_program_name(monkeypatch):
    entrypoint = importlib.import_module("corner_detector.main")
    detector = importlib.import_module("stl_tangent_boundary_demo")
    received = {}

    def fake_main(argv, *, prog=None):
        received["argv"] = argv
        received["prog"] = prog
        return "done"

    monkeypatch.setattr(detector, "main", fake_main)
    result = entrypoint.main(["--input", "model.stl", "--mode", "FPS"])

    assert result == "done"
    assert received == {
        "argv": ["--input", "model.stl", "--mode", "FPS"],
        "prog": "python -m corner_detector",
    }
