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


def test_request_for_a_pipeline_not_offered_is_rejected_before_loading_models(
    offered, monkeypatch
):
    def refuse(*args, **kwargs):
        raise AssertionError("a model was loaded for a pipeline that is not offered")

    monkeypatch.setattr(api, "APIMothDetector", refuse)
    offered("moth_binary")

    response = TestClient(api.app).post(
        "/process",
        json={"pipeline": "quebec_vermont_moths_2023", "source_images": []},
    )

    assert response.status_code == 422
    assert "not enabled" in response.json()["detail"]


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
