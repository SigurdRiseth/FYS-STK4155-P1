import pytest

from fys_stk4155_p0 import main


def test_main_prints_greeting(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    captured = capsys.readouterr()
    assert captured.out == "Hello from fys-stk4155-p0!\n"
