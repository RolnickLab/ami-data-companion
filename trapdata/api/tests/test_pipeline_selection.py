"""
Choosing which pipelines the API server and the Antenna worker offer.

Offering a pipeline loads its models, so a deployment lists the pipelines it needs in
AMI_PIPELINES and the service loads only those. These tests pin that the selection is
honoured everywhere a list of pipelines is built, and that a typo in it is caught.
None of them load a model.
"""

import json
import os
import pathlib
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import trapdata.api.api as api
from trapdata.api.schemas import PipelineConfigResponse
from trapdata.settings import Settings


@pytest.fixture
def offered(monkeypatch):
    """Set AMI_PIPELINES for the running service, as the environment would."""

    def set_pipelines(value: str):
        monkeypatch.setattr(api.settings, "pipelines", value)

    return set_pipelines


def test_setting_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("AMI_PIPELINES", "moth_binary,global_moths_2024")

    assert Settings(_env_file=None).pipelines == "moth_binary,global_moths_2024"


def test_no_selection_offers_every_pipeline(offered):
    offered("")

    assert api.select_pipelines() == api.CLASSIFIER_CHOICES


def test_setting_selects_pipelines_in_the_order_given(offered):
    offered(" global_moths_2024, moth_binary ,")

    assert list(api.select_pipelines()) == ["global_moths_2024", "moth_binary"]


def test_explicit_list_takes_precedence_over_the_setting(offered):
    offered("global_moths_2024")

    assert list(api.select_pipelines(["moth_binary"])) == ["moth_binary"]


def test_unknown_pipeline_is_rejected_with_the_valid_names(offered):
    offered("moth_binary,global_moth_2024")

    with pytest.raises(ValueError, match="global_moth_2024") as excinfo:
        api.select_pipelines()

    assert "global_moths_2024" in str(excinfo.value)


def test_service_info_loads_only_the_selected_pipelines(offered, monkeypatch):
    """
    Describing a pipeline for /info and for registration loads its models, so a
    pipeline that is not offered must not be described at all.
    """
    described = []

    def fake_pipeline_config(Classifier, slug):
        described.append(slug)
        return PipelineConfigResponse(name=slug, slug=slug, version=1)

    monkeypatch.setattr(api, "make_pipeline_config_response", fake_pipeline_config)
    offered("moth_binary")

    info = api.initialize_service_info()

    assert described == ["moth_binary"]
    assert [pipeline.slug for pipeline in info.pipelines] == ["moth_binary"]


def test_request_for_a_pipeline_not_offered_is_rejected_before_loading_models():
    """
    The request schema lists only the offered pipelines, so a request for any other
    pipeline fails validation, naming the ones on offer, before any model is built.
    """
    result = run_in_fresh_interpreter(
        "moth_binary",
        "import json\n"
        "from fastapi.testclient import TestClient\n"
        "import trapdata.api.api as api\n"
        "def refuse(*args, **kwargs):\n"
        "    raise AssertionError('a model was loaded for a pipeline not offered')\n"
        "api.APIMothDetector = refuse\n"
        "response = TestClient(api.app).post(\n"
        "    '/process',\n"
        "    json={'pipeline': 'quebec_vermont_moths_2023', 'source_images': []},\n"
        ")\n"
        "print(json.dumps({'status': response.status_code, 'body': response.json()}))\n",
    )

    assert result.returncode == 0, result.stderr
    outcome = json.loads(result.stdout.strip().splitlines()[-1])
    assert outcome["status"] == 422
    assert "moth_binary" in json.dumps(outcome["body"])


def test_registration_follows_the_settings_it_is_given(monkeypatch):
    """
    register_pipelines takes a Settings object, so it must register the pipelines
    named there rather than those in the process-wide settings.
    """
    from trapdata.antenna import registration

    described = []
    registered = []

    def fake_pipeline_config(Classifier, slug):
        described.append(slug)
        return PipelineConfigResponse(name=slug, slug=slug, version=1)

    def fake_register(*args, **kwargs):
        registered.append([config.slug for config in kwargs["pipeline_configs"]])
        return True, "registered"

    monkeypatch.setattr(api, "make_pipeline_config_response", fake_pipeline_config)
    monkeypatch.setattr(registration, "register_pipelines_for_project", fake_register)
    settings = Settings(
        _env_file=None, antenna_api_auth_token="token", pipelines="moth_binary"
    )

    registration.register_pipelines(
        project_ids=[1], service_name="Test service", settings=settings
    )

    assert described == ["moth_binary"]
    assert registered == [["moth_binary"]]


def test_registration_with_an_unknown_pipeline_stops_before_contacting_antenna(
    monkeypatch,
):
    from trapdata.antenna import registration

    def refuse(*args, **kwargs):
        raise AssertionError("Antenna was contacted despite an invalid setting")

    monkeypatch.setattr("trapdata.antenna.client.get_user_projects", refuse)
    monkeypatch.setattr(registration, "register_pipelines_for_project", refuse)
    settings = Settings(
        _env_file=None, antenna_api_auth_token="token", pipelines="not_a_pipeline"
    )

    registration.register_pipelines(
        project_ids=[], service_name="Test service", settings=settings
    )


def test_readiness_lists_only_the_offered_pipelines(offered):
    offered("moth_binary,global_moths_2024")

    response = TestClient(api.app).get("/readyz")

    assert response.json() == {"status": ["moth_binary", "global_moths_2024"]}


def test_worker_uses_the_setting_unless_pipelines_are_named(offered, monkeypatch):
    """
    The worker processes the pipelines in AMI_PIPELINES by default, the --pipeline
    option overrides the setting, and an unknown name stops the worker from starting.
    """
    from trapdata.cli.worker import cli

    started = []
    monkeypatch.setattr(
        "trapdata.antenna.worker.run_worker",
        lambda pipelines: started.append(pipelines),
    )
    offered("global_moths_2024")
    runner = CliRunner()

    assert runner.invoke(cli, []).exit_code == 0
    assert runner.invoke(cli, ["--pipeline", "moth_binary"]).exit_code == 0
    rejected = runner.invoke(cli, ["--pipeline", "not_a_pipeline"])

    assert started == [["global_moths_2024"], ["moth_binary"]]
    assert rejected.exit_code != 0
    assert "not_a_pipeline" in rejected.output


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def run_in_fresh_interpreter(
    pipelines: str, code: str
) -> subprocess.CompletedProcess[str]:
    """
    Run code against a newly imported API module with AMI_PIPELINES set.

    The request schema is built when trapdata.api.api is imported, so changing the
    setting inside this test process would not change it. A new interpreter sees the
    setting the way a deployment does when it starts.
    """
    return subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "AMI_PIPELINES": pipelines},
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_api_docs_list_only_the_offered_pipelines():
    result = run_in_fresh_interpreter(
        "moth_binary,global_moths_2024",
        "import json\n"
        "from fastapi.testclient import TestClient\n"
        "from trapdata.api.api import app\n"
        "schema = TestClient(app).get('/openapi.json').json()\n"
        "print(json.dumps(schema['components']['schemas']['PipelineChoice']['enum']))\n",
    )

    assert result.returncode == 0, result.stderr
    offered = json.loads(result.stdout.strip().splitlines()[-1])
    assert offered == ["moth_binary", "global_moths_2024"]


def test_invalid_setting_keeps_imports_working_but_stops_the_server():
    """
    Every ami command imports the API module, so a typo in AMI_PIPELINES must not
    break the import, while the API server must still refuse to start.
    """
    result = run_in_fresh_interpreter(
        "moth_binary,not_a_pipeline",
        "from fastapi.testclient import TestClient\n"
        "from trapdata.api.api import app\n"
        "print('imported')\n"
        "with TestClient(app):\n"
        "    pass\n",
    )

    assert "imported" in result.stdout
    assert result.returncode != 0
    assert "not_a_pipeline" in result.stderr


def test_api_command_exits_on_an_invalid_setting_before_starting_the_server(
    offered, monkeypatch
):
    """
    `ami api` runs uvicorn with its reloader, which keeps running when the app fails
    at startup, so the command itself must reject the setting and exit.
    """
    from trapdata.cli.base import cli

    def refuse(*args, **kwargs):
        raise AssertionError("uvicorn was started despite an invalid setting")

    monkeypatch.setattr("uvicorn.run", refuse)
    offered("moth_binary,not_a_pipeline")

    result = CliRunner().invoke(cli, ["api"])

    assert isinstance(result.exception, SystemExit)
    assert result.exit_code != 0
    assert "not_a_pipeline" in result.output


def test_register_command_fails_on_an_invalid_setting(offered, monkeypatch):
    """
    An invalid setting makes `ami worker register` exit with an error, so a script
    or CI job does not read the run as a success.
    """
    from trapdata.cli.worker import cli

    def refuse(*args, **kwargs):
        raise AssertionError("registration ran despite an invalid setting")

    monkeypatch.setattr("trapdata.antenna.registration.register_pipelines", refuse)
    offered("not_a_pipeline")

    result = CliRunner().invoke(cli, ["register"])

    assert isinstance(result.exception, SystemExit)
    assert result.exit_code != 0
    assert "not_a_pipeline" in result.output
