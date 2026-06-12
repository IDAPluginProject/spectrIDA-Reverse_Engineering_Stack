from spectrida import config


def test_env_override(monkeypatch):
    monkeypatch.setenv("SPECTRIDA_MODEL", "my-model")
    assert config.ollama_model() == "my-model"


def test_defaults():
    assert config.ollama_url().startswith("http")
    assert config.pipeline_workers() == 16


def test_onboarded_marker(tmp_path, monkeypatch):
    monkeypatch.delenv("SPECTRIDA_NO_ONBOARD", raising=False)
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "_ONBOARD_MARKER", tmp_path / ".onboarded")
    assert config.onboarded() is False
    config.set_onboarded()
    assert config.onboarded() is True


def test_env_forces_skip(monkeypatch):
    monkeypatch.setenv("SPECTRIDA_NO_ONBOARD", "1")
    assert config.onboarded() is True
