"""Unit tests for the staged pipeline registry and its execution.

These tests are deliberately weight-free: they stub the model stages so nothing
loads real detector or classifier weights, keeping them fast and runnable on a
machine without a GPU. They pin four properties that are invisible in a diff and
have regressed before:

* the pipelines the service advertises, the worker subscribes to, and job
  dispatch accepts are the same set (no advertise-but-dead / execute-but-hidden
  drift);
* the staged ``/process`` execution assembles results by detection identity, so
  a gate-dropped or un-croppable detection keeps its own label instead of
  shifting its neighbours';
* a passing detection carries both its (non-terminal) gate label and its
  terminal label, so the platform never shows the gate label as the best result;
* the over-confidence ``score_scale`` tempers reported scores only, leaving the
  stored logits honest.
"""

import datetime
from unittest import mock

import pytest
import torch

from trapdata.api import api
from trapdata.api.api import (
    PIPELINE_CHOICES,
    IntermediateStage,
    PipelineDefinition,
    run_classification_stages,
)
from trapdata.api.models.classification import (
    APIMothClassifier,
    InsectOrderClassifier,
    MothClassifierGlobal,
)
from trapdata.api.models.localization import APIAnyBugDetector, APIMothDetector
from trapdata.api.schemas import (
    AlgorithmReference,
    BoundingBox,
    ClassificationResponse,
    DetectionResponse,
    PipelineConfigResponse,
)
from trapdata.cli.worker import default_pipeline_slugs

# ---------------------------------------------------------------------------
# Weight-free stub stages
# ---------------------------------------------------------------------------


def _make_detection(detection_id: str) -> DetectionResponse:
    return DetectionResponse(
        source_image_id=detection_id,
        bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1),
        algorithm=AlgorithmReference(name="stub-detector", key="stub_detector"),
        timestamp=datetime.datetime.now(),
    )


class _StubClassifierStage:
    """Stand-in for an ``APIMothClassifier`` stage that annotates the detections
    it is handed in place, exactly as the real ``save_results`` does, without
    loading any weights.

    ``task_type`` is set so the registry's shape validation accepts it as a
    classification stage. Subclasses configure ``algorithm_key`` and the
    per-detection ``labels`` map, and may list ``drop_ids`` to simulate crops the
    model could not read (those detections stay in ``results`` un-annotated).
    """

    task_type = "classification"
    name = "stub-classifier"
    description = None
    weights_path = None
    category_map: dict = {}  # falsy -> make_algorithm_response emits no category map

    algorithm_key = "stub_classifier"
    labels: dict[str, str] = {}
    drop_ids: frozenset[str] = frozenset()

    def __init__(self, source_images=None, detections=(), terminal=True, **kwargs):
        self.detections = list(detections)
        self.terminal = terminal
        self.results: list[DetectionResponse] = []

    def get_key(self) -> str:
        return self.algorithm_key

    def run(self) -> list[DetectionResponse]:
        for detection in self.detections:
            if detection.source_image_id in self.drop_ids:
                # Un-croppable detection: left un-annotated but still returned,
                # mirroring the real classifier whose results == its detections.
                continue
            label = self.labels[detection.source_image_id]
            detection.classifications = [
                c
                for c in detection.classifications
                if c.algorithm.key != self.get_key()
            ]
            detection.classifications.append(
                ClassificationResponse(
                    classification=label,
                    scores=[1.0],
                    logits=[0.0],
                    algorithm=AlgorithmReference(name=self.name, key=self.get_key()),
                    terminal=self.terminal,
                    timestamp=datetime.datetime.now(),
                )
            )
        self.results = self.detections
        return self.results


def _classification_for(detection: DetectionResponse, key: str):
    for classification in detection.classifications:
        if classification.algorithm.key == key:
            return classification
    return None


# ---------------------------------------------------------------------------
# Item 3: advertised == subscribed == dispatchable
# ---------------------------------------------------------------------------


def test_advertised_subscribed_and_dispatchable_slugs_match():
    """Every pipeline in the registry is advertised by /info, subscribed to by a
    default worker, and dispatchable by the worker's job lookup — with no slug
    present in one view but missing from another.

    ``make_pipeline_config_response`` is stubbed so ``initialize_service_info``
    builds a config for every pipeline without loading weights (otherwise a
    pipeline whose weight is absent, such as the anybug detector, would silently
    drop out of /info and mask exactly the drift this test guards against).
    """

    def _fake_config(pipeline, slug):
        return PipelineConfigResponse(name=slug, slug=slug, version=1, algorithms=[])

    with mock.patch.object(api, "make_pipeline_config_response", _fake_config):
        advertised = {p.slug for p in api.initialize_service_info().pipelines}

    subscribed = set(default_pipeline_slugs())
    dispatchable = set(PIPELINE_CHOICES.keys())

    assert advertised == subscribed == dispatchable
    # The regression that motivated this test: the new anybug pipeline must be in
    # all three sets, not advertised-but-undispatchable.
    assert "anybug_global_moths_2024" in advertised


# ---------------------------------------------------------------------------
# Advertised algorithm list: exactly one detector per pipeline
# ---------------------------------------------------------------------------
#
# Antenna's save_results rejects any pipeline whose /info advertises more than one
# localization ("detection") algorithm, so detections never save. A pre-refactor
# builder hardcoded the FasterRCNN moth detector for every pipeline; layering the
# anybug pipeline (which has its own YOLO26 detector) on top advertised TWO
# detectors. The typed PipelineDefinition now sources the detector from a single
# validated field, and make_pipeline_config_response emits one algorithm per
# stage. These tests pin both halves so the double-detector regression cannot
# return, and are weight-free (they read class task_types / stub the stages).
#
# Only the builder test below exercises make_pipeline_config_response itself, so
# it is the one that fails if a default detector is reintroduced there; the other
# two pin what the registry declares. Each docstring says which half it covers.


def _advertised_stage_classes(pipeline: PipelineDefinition) -> list[type]:
    """The model classes make_pipeline_config_response emits one algorithm for, in
    the order it emits them: the detector, each intermediate gate, then the
    terminal classifier.
    """
    return [
        pipeline.detector,
        *(stage.classifier for stage in pipeline.intermediates),
        pipeline.terminal,
    ]


@pytest.mark.parametrize("slug", list(PIPELINE_CHOICES))
def test_every_pipeline_advertises_exactly_one_localization_algorithm(slug):
    """Each registered pipeline declares exactly one localization (detector)
    stage — the count Antenna's save_results allows.

    PipelineDefinition already enforces this by construction, since it validates
    that ``detector`` holds a localization model and every other slot holds a
    classification model. This test therefore guards that validation rather than
    the registry entries: it fails if a later change relaxes the roles or adds a
    second detector slot. The builder half of the invariant, that nothing injects
    an extra detector while assembling the response, is covered separately by
    test_make_pipeline_config_response_emits_one_algorithm_per_stage.
    """
    stages = _advertised_stage_classes(PIPELINE_CHOICES[slug])
    localization = [c for c in stages if c.task_type == "localization"]
    assert len(localization) == 1, (
        f"{slug} advertises detectors {[c.__name__ for c in localization]}; "
        "exactly one localization algorithm is allowed"
    )


def test_anybug_advertises_its_yolo26_detector_not_fasterrcnn():
    """The anybug pipeline advertises its own YOLO26 detector as the sole
    localization algorithm, not the FasterRCNN moth detector a default-detector
    builder would inject, and keeps its two classification stages intact.
    """
    stages = _advertised_stage_classes(PIPELINE_CHOICES["anybug_global_moths_2024"])
    localization = [c for c in stages if c.task_type == "localization"]
    assert localization == [APIAnyBugDetector]
    assert localization[0].get_key() == "anybug-yolo26x-detector-2024"
    # The spurious FasterRCNN detector must not appear anywhere in the stage list.
    assert APIMothDetector not in stages
    classification = [c for c in stages if c.task_type == "classification"]
    assert len(classification) == 2


# Stand-in stage models, so the builder can be exercised without downloading any
# weights. They carry only the attributes make_algorithm_config_response reads.


class _ConfigBuilderDetector:
    task_type = "localization"
    name = "Stub Detector"
    description = "stub detector"
    weights_path = "http://example.invalid/detector.pt"
    labels_path = None
    default_taxon_rank = "SPECIES"
    category_map = {0: "object"}

    def __init__(self, source_images=(), **kwargs):
        pass

    def get_key(self) -> str:
        return "stub-detector"


class _ConfigBuilderGate:
    task_type = "classification"
    name = "Stub Order Gate"
    description = "stub gate"
    weights_path = "http://example.invalid/gate.pt"
    labels_path = None
    default_taxon_rank = "ORDER"
    category_map = {0: "Lepidoptera", 1: "Coleoptera"}

    def __init__(self, source_images=(), detections=(), terminal=True, **kwargs):
        pass

    def get_key(self) -> str:
        return "stub-order-gate"


class _ConfigBuilderTerminal:
    task_type = "classification"
    name = "Stub Species Classifier"
    description = "stub terminal"
    weights_path = "http://example.invalid/species.pt"
    labels_path = None
    default_taxon_rank = "SPECIES"
    category_map = {0: "Species A", 1: "Species B"}

    def __init__(self, source_images=(), detections=(), terminal=True, **kwargs):
        pass

    def get_key(self) -> str:
        return "stub-species"


def _stub_pipeline(**kwargs) -> PipelineDefinition:
    """A three-stage pipeline of stub models: detector, one gate, terminal."""
    return PipelineDefinition(
        detector=_ConfigBuilderDetector,
        terminal=_ConfigBuilderTerminal,
        intermediates=(
            IntermediateStage(
                classifier=_ConfigBuilderGate, pass_labels=("Lepidoptera",)
            ),
        ),
        **kwargs,
    )


def test_make_pipeline_config_response_emits_one_algorithm_per_stage():
    """The /info builder emits exactly one algorithm per configured stage, in
    stage order, and injects no extra detector of its own. This guards the builder
    itself against re-introducing a hardcoded/default detector union (the shape of
    the deployed regression), independently of the registry entries.
    """
    config = api.make_pipeline_config_response(_stub_pipeline(), slug="stub_pipeline")

    task_types = [algorithm.task_type for algorithm in config.algorithms]
    assert task_types == ["localization", "classification", "classification"]
    localization = [a for a in config.algorithms if a.task_type == "localization"]
    assert [a.key for a in localization] == ["stub-detector"]


# ---------------------------------------------------------------------------
# Advertised pipeline name: distinct per pipeline
# ---------------------------------------------------------------------------
#
# The platform lists pipelines by the name /info advertises, so two pipelines
# sharing a name are indistinguishable to the operator picking one. The name
# falls back to the terminal classifier's, which stayed unique only while every
# pipeline had a terminal classifier to itself.


def test_every_pipeline_advertises_a_distinct_name():
    """No two registered pipelines advertise the same name.

    Both "global_moths_2024" and "anybug_global_moths_2024" end in
    MothClassifierGlobal, so before the anybug pipeline was given a name of its
    own they both advertised "Global Species Classifier - Aug 2024" and appeared
    as two identical entries. Reads class attributes only, so no weights load.
    """
    slug_by_name: dict[str, str] = {}
    for slug, pipeline in PIPELINE_CHOICES.items():
        advertised = pipeline.name or pipeline.terminal.name
        assert advertised not in slug_by_name, (
            f"{slug!r} and {slug_by_name[advertised]!r} both advertise the name "
            f"{advertised!r}; set PipelineDefinition.name on one of them"
        )
        slug_by_name[advertised] = slug


def test_pipeline_config_name_prefers_the_pipeline_over_its_terminal():
    """The builder advertises the pipeline's own name and description when it has
    them, and falls back to the terminal classifier's when it does not, so adding
    the fields left the pipelines that omit them unchanged.
    """
    named = api.make_pipeline_config_response(
        _stub_pipeline(name="Custom Name", description="Custom description"),
        slug="stub_pipeline",
    )
    assert named.name == "Custom Name"
    assert named.description == "Custom description"

    unnamed = api.make_pipeline_config_response(_stub_pipeline(), slug="stub_pipeline")
    assert unnamed.name == _ConfigBuilderTerminal.name
    assert unnamed.description == _ConfigBuilderTerminal.description


# ---------------------------------------------------------------------------
# Items 4 & 5: identity-keyed assembly, gate vs terminal labelling
# ---------------------------------------------------------------------------


def test_gate_merge_keeps_labels_by_detection_identity():
    """A gate that drops detections in the MIDDLE of the batch, followed by a
    terminal classifier that cannot crop one of the passing detections, must
    leave every detection with its own correct label.

    This is the property an index-based (zip-by-position) assembly would break:
    after a dropped item, later detections would inherit their neighbours'
    labels. The assembly keys on detection identity instead, so this pins that.
    """

    class _Gate(_StubClassifierStage):
        algorithm_key = "order_gate"
        name = "Order Gate"
        labels = {
            "d0": "Lepidoptera",  # passes
            "d1": "Coleoptera",  # fails in the middle
            "d2": "Lepidoptera",  # passes
            "d3": "Diptera",  # fails in the middle
            "d4": "Lepidoptera",  # passes
        }

    class _Terminal(_StubClassifierStage):
        algorithm_key = "species"
        name = "Species Classifier"
        labels = {"d0": "Species A", "d2": "Species C", "d4": "Species E"}
        drop_ids = frozenset({"d2"})  # a passing detection whose crop fails

    pipeline = PipelineDefinition(
        detector=APIMothDetector,  # never instantiated here; results passed in
        terminal=_Terminal,
        intermediates=(
            IntermediateStage(classifier=_Gate, pass_labels=("Lepidoptera",)),
        ),
    )

    detections = [_make_detection(f"d{i}") for i in range(5)]
    algorithms_used: dict = {}

    returned = run_classification_stages(
        pipeline=pipeline,
        source_images=[],
        detector_results=detections,
        example_config_param=None,
        algorithms_used=algorithms_used,
    )

    # Every detection is returned exactly once, none lost or duplicated.
    returned_ids = [d.source_image_id for d in returned]
    assert sorted(returned_ids) == ["d0", "d1", "d2", "d3", "d4"]
    assert len(returned_ids) == len(set(returned_ids))

    by_id = {d.source_image_id: d for d in returned}

    # Non-passing detections keep their own gate label (non-terminal) and get no
    # terminal (species) classification.
    for did, order in (("d1", "Coleoptera"), ("d3", "Diptera")):
        gate_c = _classification_for(by_id[did], "order_gate")
        assert gate_c is not None and gate_c.classification == order
        assert gate_c.terminal is False
        assert _classification_for(by_id[did], "species") is None

    # Passing + classified detections carry BOTH the non-terminal gate label and
    # their own terminal species label — the correct one for that identity.
    for did, species in (("d0", "Species A"), ("d4", "Species E")):
        gate_c = _classification_for(by_id[did], "order_gate")
        assert gate_c is not None and gate_c.classification == "Lepidoptera"
        assert gate_c.terminal is False
        species_c = _classification_for(by_id[did], "species")
        assert species_c is not None and species_c.classification == species
        assert species_c.terminal is True

    # The passing detection whose crop the terminal could not read keeps its gate
    # label and is returned, but is NOT mislabelled with a neighbour's species.
    d2_gate = _classification_for(by_id["d2"], "order_gate")
    assert d2_gate is not None and d2_gate.classification == "Lepidoptera"
    assert _classification_for(by_id["d2"], "species") is None


def test_passing_detection_marks_only_terminal_stage_terminal():
    """For a detection that passes the gate, the gate classification is marked
    non-terminal and the terminal classification is marked terminal, so the
    platform selects the species label as the best result rather than the gate.
    """

    class _Gate(_StubClassifierStage):
        algorithm_key = "order_gate"
        labels = {"only": "Lepidoptera"}

    class _Terminal(_StubClassifierStage):
        algorithm_key = "species"
        labels = {"only": "Some species"}

    pipeline = PipelineDefinition(
        detector=APIMothDetector,
        terminal=_Terminal,
        intermediates=(
            IntermediateStage(classifier=_Gate, pass_labels=("Lepidoptera",)),
        ),
    )

    [detection] = run_classification_stages(
        pipeline=pipeline,
        source_images=[],
        detector_results=[_make_detection("only")],
        example_config_param=None,
        algorithms_used={},
    )

    terminal_labels = [
        c.classification for c in detection.classifications if c.terminal
    ]
    non_terminal_labels = [
        c.classification for c in detection.classifications if not c.terminal
    ]
    assert terminal_labels == ["Some species"]
    assert non_terminal_labels == ["Lepidoptera"]


# ---------------------------------------------------------------------------
# Item 6: score_scale tempers scores only, logits stay honest
# ---------------------------------------------------------------------------


def test_score_scale_scales_scores_not_logits():
    """``post_process_batch`` multiplies the reported softmax scores by
    ``score_scale`` but stores the raw, unscaled logits, so the true confidence
    stays recoverable even when a chronically over-confident model is tempered.
    """
    logits = torch.tensor([[2.0, 0.0, -1.0]])
    expected_softmax = torch.nn.functional.softmax(logits, dim=1)[0].tolist()

    fake_self = mock.Mock()
    fake_self.category_map = {0: "a", 1: "b", 2: "c"}
    fake_self.score_scale = 0.5

    [result] = APIMothClassifier.post_process_batch(fake_self, logits)

    # Logits are stored raw (not multiplied by score_scale).
    assert result.logit == pytest.approx(logits[0].tolist())
    # Scores are the softmax tempered by score_scale.
    assert result.scores == pytest.approx([s * 0.5 for s in expected_softmax])
    # Tempering keeps argmax invariant: the top class is unchanged.
    assert max(range(3), key=lambda i: result.scores[i]) == 0


def test_order_gate_keeps_product_score_scale():
    """The Lepidoptera order gate is the InsectOrderClassifier, whose reported
    scores are capped below certainty by an explicit product choice. Guard the
    constant so a refactor does not quietly reset it to 1.0.
    """
    assert InsectOrderClassifier.score_scale == 0.9
    assert api.LEPIDOPTERA_ORDER_GATE.classifier is InsectOrderClassifier
    # The gate does not change which pipeline is terminal.
    assert PIPELINE_CHOICES["anybug_global_moths_2024"].terminal is MothClassifierGlobal


# ---------------------------------------------------------------------------
# Item 7: stage dataclasses validate role/shape on construction
# ---------------------------------------------------------------------------


def test_pipeline_definition_rejects_wrong_roles():
    """The registry validates each slot's role on construction rather than
    trusting the caller: a detector cannot be used where a classifier belongs
    (or vice versa), and intermediates must be IntermediateStage instances.
    """
    with pytest.raises(TypeError):
        PipelineDefinition(detector=MothClassifierGlobal, terminal=MothClassifierGlobal)

    with pytest.raises(TypeError):
        PipelineDefinition(detector=APIMothDetector, terminal=APIMothDetector)

    with pytest.raises(TypeError):
        PipelineDefinition(
            detector=APIMothDetector,
            terminal=MothClassifierGlobal,
            intermediates=(APIMothDetector,),  # not an IntermediateStage
        )

    with pytest.raises(TypeError):
        # A detector cannot fill an intermediate gate slot.
        IntermediateStage(classifier=APIMothDetector)

    with pytest.raises(TypeError):
        IntermediateStage(classifier=MothClassifierGlobal, pass_labels=["not", "tuple"])


# ---------------------------------------------------------------------------
# Item 1: the worker's job dispatch reads the same registry
# ---------------------------------------------------------------------------


def test_worker_dispatch_runs_anybug_without_notimplemented():
    """The worker is now stage-driven, so the anybug pipeline dispatches through
    the same path as the legacy pipelines instead of hitting a capability guard.

    With an empty job the worker completes (no work) without raising
    NotImplementedError and without constructing any stage model — the YOLO26
    detector and order-gate weights are only loaded when there is data to
    process, which keeps this test weight-free. This pins that the old
    "advertised but undispatchable" guard is gone.
    """
    from trapdata.antenna import worker

    with mock.patch.object(worker, "get_rest_dataloader", return_value=iter([])):
        with mock.patch("torch.cuda.is_available", return_value=False):
            did_work = worker._process_job(
                "anybug_global_moths_2024", job_id=1, settings=mock.Mock()
            )
    assert did_work is False


def test_worker_dispatch_accepts_legacy_pipeline_unchanged():
    """A legacy moth pipeline still dispatches through the migrated lookup with no
    behavior change: given an empty job the worker completes without loading any
    model weights.
    """
    from trapdata.antenna import worker

    with mock.patch.object(worker, "get_rest_dataloader", return_value=iter([])):
        with mock.patch("torch.cuda.is_available", return_value=False):
            did_work = worker._process_job(
                "quebec_vermont_moths_2023", job_id=1, settings=mock.Mock()
            )
    assert did_work is False


# ---------------------------------------------------------------------------
# The worker's _process_batch is stage-driven (weight-free)
# ---------------------------------------------------------------------------
#
# These stubs stand in for the pipeline's detector and classifier stages so the
# worker's async inference core (_process_batch -> run_classification_stages)
# runs end-to-end without loading any weights. Each stage assigns labels by
# detection identity in update_detection_classification, so the crop tensors are
# irrelevant to the label a detection ends up with.


class _StubDetector:
    """Weight-free stand-in for the pipeline's detector. Emits one full-frame
    bounding box per image, mirroring APIMothDetector/APIAnyBugDetector's
    predict/post-process/save contract over in-memory image tensors.
    """

    task_type = "localization"
    name = "stub-detector"

    def __init__(self, source_images=(), **kwargs):
        self.results: list[DetectionResponse] = []

    def reset(self, source_images):
        self.results = []

    def predict_batch(self, images):
        return list(images)

    def post_process_batch(self, batch_output):
        # One box per image spanning the whole (16x16) stub tensor.
        return [[[0, 0, 16, 16]] for _ in batch_output]

    def save_results(self, item_ids, batch_output, seconds_per_item, *args, **kwargs):
        for image_id, image_output in zip(item_ids, batch_output):
            for coords in image_output:
                self.results.append(
                    DetectionResponse(
                        source_image_id=image_id,
                        bbox=BoundingBox(
                            x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3]
                        ),
                        algorithm=AlgorithmReference(
                            name=self.name, key="stub_detector"
                        ),
                        timestamp=datetime.datetime.now(),
                    )
                )


class _StubTensorStage:
    """Weight-free stand-in for an APIMothClassifier stage as the worker drives
    it: reset -> get_transforms -> predict_batch -> post_process_batch ->
    update_detection_classification. Labels come from ``labels`` keyed by
    ``source_image_id`` so the assigned label does not depend on crop content.
    """

    task_type = "classification"
    name = "stub-stage"
    description = None
    weights_path = None
    category_map: dict = {}

    algorithm_key = "stub_stage"
    labels: dict[str, str] = {}

    def __init__(self, source_images=(), detections=(), terminal=True, **kwargs):
        self.detections = list(detections)
        self.terminal = terminal
        self.results: list[DetectionResponse] = []

    def get_key(self) -> str:
        return self.algorithm_key

    def reset(self, detections):
        self.detections = list(detections)
        self.results = []

    def get_transforms(self):
        # Resize every crop to a fixed shape so torch.stack succeeds regardless
        # of bbox size; the label is applied by identity, not crop content.
        return lambda crop: torch.zeros(3, 4, 4)

    def predict_batch(self, batched_crops):
        return batched_crops

    def post_process_batch(self, batch_output):
        # One placeholder prediction per crop; the real label is applied in
        # update_detection_classification by detection identity.
        return [None for _ in batch_output]

    def update_detection_classification(
        self, seconds_per_item, image_id, detection_idx, predictions
    ):
        detection = self.detections[detection_idx]
        detection.classifications = [
            c for c in detection.classifications if c.algorithm.key != self.get_key()
        ]
        detection.classifications.append(
            ClassificationResponse(
                classification=self.labels[detection.source_image_id],
                scores=[1.0],
                logits=[0.0],
                algorithm=AlgorithmReference(name=self.name, key=self.get_key()),
                terminal=self.terminal,
                timestamp=datetime.datetime.now(),
            )
        )
        return detection


def _make_stub_batch(image_ids: list[str]) -> dict:
    return {
        "images": [torch.zeros(3, 16, 16) for _ in image_ids],
        "image_ids": list(image_ids),
        "reply_subjects": [f"reply_{i}" for i in image_ids],
        "image_urls": [f"http://example.com/{i}.jpg" for i in image_ids],
        "failed_items": [],
    }


def _classifications_by_algorithm(detection: DetectionResponse) -> dict:
    return {
        c.algorithm.key: (c.classification, c.terminal)
        for c in detection.classifications
    }


def test_process_batch_runs_anybug_stage_list_and_posts_labels():
    """The stage-driven worker runs the anybug stage list (detector -> order gate
    -> terminal species) over in-memory tensors without NotImplementedError, and
    posts each detection labelled by the stage that owns it: a gate-dropped
    detection carries only its non-terminal order label, a passing detection
    carries both the non-terminal order label and its terminal species label.
    """
    from trapdata.antenna import worker

    class _OrderGate(_StubTensorStage):
        algorithm_key = "order_gate"
        name = "Order Gate"
        labels = {"img0": "Lepidoptera", "img1": "Coleoptera", "img2": "Lepidoptera"}

    class _Species(_StubTensorStage):
        algorithm_key = "species"
        name = "Species Classifier"
        labels = {"img0": "Species A", "img2": "Species B"}  # only the passers

    pipeline_def = PipelineDefinition(
        detector=_StubDetector,
        terminal=_Species,
        intermediates=(
            IntermediateStage(classifier=_OrderGate, pass_labels=("Lepidoptera",)),
        ),
    )

    image_ids = ["img0", "img1", "img2"]
    n_items, n_detections, batch_results, _, _ = worker._process_batch(
        _make_stub_batch(image_ids),
        0,
        _StubDetector(),
        worker._build_stage_models(pipeline_def),
        pipeline_def,
        "anybug_global_moths_2024",
    )

    assert n_items == 3
    # Every detection is posted exactly once: 1 gate-dropped + 2 terminal.
    assert n_detections == 3
    assert len(batch_results) == 3

    by_subject = {r.reply_subject: r.result for r in batch_results}
    assert by_subject["reply_img0"].pipeline == "anybug_global_moths_2024"

    # img0 passes the order gate: non-terminal Lepidoptera + terminal species.
    [d0] = by_subject["reply_img0"].detections
    assert _classifications_by_algorithm(d0) == {
        "order_gate": ("Lepidoptera", False),
        "species": ("Species A", True),
    }

    # img1 fails the order gate: only the non-terminal order label, no species.
    [d1] = by_subject["reply_img1"].detections
    assert _classifications_by_algorithm(d1) == {"order_gate": ("Coleoptera", False)}


def test_process_batch_legacy_binary_split_preserved():
    """A legacy binary pipeline still splits moth vs non-moth exactly: non-moth
    detections are returned tagged with the non-terminal binary label and no
    species, while moth detections carry the non-terminal binary label plus the
    terminal species label. This is the behavior-preserving guarantee for the
    detector+binary-gate+terminal pipelines now that they run through the shared
    stage engine.
    """
    from trapdata.antenna import worker

    class _Binary(_StubTensorStage):
        algorithm_key = "binary"
        name = "Moth / Non-Moth"
        labels = {"img0": "moth", "img1": "nonmoth", "img2": "moth"}

    class _Species(_StubTensorStage):
        algorithm_key = "species"
        name = "Species Classifier"
        labels = {"img0": "Species A", "img2": "Species B"}

    pipeline_def = PipelineDefinition(
        detector=_StubDetector,
        terminal=_Species,
        intermediates=(IntermediateStage(classifier=_Binary, pass_labels=("moth",)),),
    )

    image_ids = ["img0", "img1", "img2"]
    n_items, n_detections, batch_results, _, _ = worker._process_batch(
        _make_stub_batch(image_ids),
        0,
        _StubDetector(),
        worker._build_stage_models(pipeline_def),
        pipeline_def,
        "quebec_vermont_moths_2023",
    )

    assert n_items == 3
    assert n_detections == 3
    by_subject = {r.reply_subject: r.result for r in batch_results}

    # Non-moth is returned tagged non-terminal binary, no species (the split is
    # preserved: it never reaches the terminal classifier).
    [d1] = by_subject["reply_img1"].detections
    assert _classifications_by_algorithm(d1) == {"binary": ("nonmoth", False)}

    # Moth carries the non-terminal binary label plus the terminal species label.
    [d0] = by_subject["reply_img0"].detections
    assert _classifications_by_algorithm(d0) == {
        "binary": ("moth", False),
        "species": ("Species A", True),
    }


def test_process_batch_tags_uncroppable_detection_instead_of_naked():
    """A detection whose bbox is degenerate (here zero-width) is returned TAGGED
    with the non-terminal 'uncroppable' sentinel rather than naked, so the
    pipeline's "every detection carries a classification" invariant holds. The
    box never reaches the gate or terminal label, mirroring how a gate tags the
    detections it drops.
    """
    from trapdata.antenna import worker

    class _DegenerateDetector(_StubDetector):
        def post_process_batch(self, batch_output):
            # x1 == x2 -> zero-width box the crop loop cannot slice.
            return [[[5, 5, 5, 12]] for _ in batch_output]

    class _OrderGate(_StubTensorStage):
        algorithm_key = "order_gate"
        name = "Order Gate"
        labels = {"img0": "Lepidoptera"}  # never applied: the crop is skipped

    class _Species(_StubTensorStage):
        algorithm_key = "species"
        name = "Species Classifier"
        labels: dict[str, str] = {}

    pipeline_def = PipelineDefinition(
        detector=_DegenerateDetector,
        terminal=_Species,
        intermediates=(
            IntermediateStage(classifier=_OrderGate, pass_labels=("Lepidoptera",)),
        ),
    )

    _, n_detections, batch_results, _, _ = worker._process_batch(
        _make_stub_batch(["img0"]),
        0,
        _DegenerateDetector(),
        worker._build_stage_models(pipeline_def),
        pipeline_def,
        "anybug_global_moths_2024",
    )

    assert n_detections == 1
    [d0] = batch_results[0].result.detections
    # Tagged with the non-terminal uncroppable sentinel, and NOTHING else: no
    # gate label, no species label, and not a naked (classification-free) box.
    assert _classifications_by_algorithm(d0) == {
        worker.UNCROPPABLE_ALGORITHM.key: (worker.UNCROPPABLE_LABEL, False)
    }
