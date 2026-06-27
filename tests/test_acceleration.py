from pdiseg.core.acceleration import acceleration_info, gpu_modules, requested_backend
from pdiseg.runtime.env import env_int


def test_requested_backend_accepts_cpu_alias(monkeypatch):
    monkeypatch.setenv("PDISEG_BACKEND", "off")

    assert requested_backend() == "cpu"


def test_acceleration_info_respects_forced_cpu(monkeypatch):
    monkeypatch.setenv("PDISEG_BACKEND", "cpu")

    info = acceleration_info()

    assert info.backend == "cpu"
    assert info.requested == "cpu"


def test_gpu_modules_respects_forced_cpu(monkeypatch):
    monkeypatch.setenv("PDISEG_BACKEND", "cpu")

    assert gpu_modules() is None


def test_env_int_reads_valid_integer(monkeypatch):
    monkeypatch.setenv("PDISEG_WORKERS", "4")

    assert env_int("PDISEG_WORKERS", 1) == 4


def test_env_int_falls_back_on_blank_or_invalid(monkeypatch):
    monkeypatch.setenv("PDISEG_WORKERS", "")
    assert env_int("PDISEG_WORKERS", 3) == 3

    monkeypatch.setenv("PDISEG_WORKERS", "many")
    assert env_int("PDISEG_WORKERS", 3) == 3
