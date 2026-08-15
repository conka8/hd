"""Canonical Bench v9 confirmation evidence and Go adapter guards."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ditto_screening_protocol.confirmation import (
    CANONICAL_CONFIRMATION_ROOT_FIELDS,
    AblationEvidence,
    ConfirmationCompletionReport,
    ConfirmationEvidenceRoot,
    V9ConfirmationEvidenceRoot,
    canonical_json,
    evidence_digest,
)
from ditto_screening_protocol.confirmation_wire import (
    ConfirmationWireError,
    completion_report_from_go_fixture,
    longmem_envelope_from_go,
)

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[3]
    / "services"
    / "dittobench-api"
    / "internal"
    / "confirmationwire"
    / "testdata"
    / "go_confirmation_evidence_v9.json"
)
_ZERO_LONGMEM_FIXTURE_PATH = (
    Path(__file__).resolve().parents[3]
    / "services"
    / "dittobench-api"
    / "internal"
    / "longmemeval"
    / "testdata"
    / "go_longmem_official_zero_v2.json"
)


def _fixture() -> dict[str, object]:
    return json.loads(_FIXTURE_PATH.read_text())


def _report() -> ConfirmationCompletionReport:
    return completion_report_from_go_fixture(
        _fixture(),
        ablation_coordinator_latency_ms=444,
    )


def test_signed_root_field_order_is_one_exact_shared_constant() -> None:
    expected = (
        "schema_version",
        "artifact_sha256",
        "bench_version",
        "confirmation_profile_revision",
        "confirmation_profile_checksum",
        "settings_revision",
        "settings_checksum",
        "retest_generation",
        "ablation_coordinator_latency_ms",
        "composite_policy",
        "longmemeval",
        "inference_ablation",
        "embedding_ablation",
        "totals",
    )
    assert expected == CANONICAL_CONFIRMATION_ROOT_FIELDS
    assert tuple(ConfirmationEvidenceRoot.model_fields) == expected
    assert tuple(V9ConfirmationEvidenceRoot.model_fields) == expected


def test_real_go_fixture_has_pinned_normalized_canonical_digests() -> None:
    report = _report()

    assert evidence_digest(report.longmemeval.evidence) == (
        "6a780e89db47149ea6531fc43a6f59208a880983dbab6e526da0ea63307980be"
    )
    assert report.longmemeval.evidence_sha256 == evidence_digest(
        report.longmemeval.evidence
    )
    assert report.inference_ablation.evidence_sha256 == evidence_digest(
        report.inference_ablation.evidence
    )
    assert report.embedding_ablation.evidence_sha256 == evidence_digest(
        report.embedding_ablation.evidence
    )
    assert hashlib.sha256(canonical_json(report)).hexdigest() == (
        "3266ec7830fa0dda44f28e86ec649f9a8b45d7073a3552d6312303a10dd5aa26"
    )


def test_real_go_official_zero_fixture_normalizes_without_synthetic_receipts() -> None:
    raw = json.loads(_ZERO_LONGMEM_FIXTURE_PATH.read_text())
    envelope = longmem_envelope_from_go(raw)

    assert envelope.request_count == 0
    assert envelope.input_tokens == 0
    assert envelope.output_tokens == 0
    assert envelope.provider_cost_microusd == 0
    assert envelope.evidence.score.case_count == 12
    assert envelope.evidence.score.longmem_mean_micros == 0
    assert envelope.evidence.score.longmem_stderr_micros == 0
    assert all(row.correct == 0 for row in envelope.evidence.score.per_capability)
    assert all(row.requests == 0 for row in envelope.evidence.provider_evidence)
    assert all(
        row.receipt_set_sha256 == "" for row in envelope.evidence.provider_evidence
    )


def test_official_zero_requires_explicit_empty_receipt_and_rejects_mixed_lanes() -> (
    None
):
    raw = json.loads(_ZERO_LONGMEM_FIXTURE_PATH.read_text())
    evidence = raw["evidence"]
    assert isinstance(evidence, dict)
    providers = evidence["provider_evidence"]
    assert isinstance(providers, list)
    first = providers[0]
    assert isinstance(first, dict)
    del first["receipt_set_sha256"]
    with pytest.raises(ConfirmationWireError, match="fields drifted"):
        longmem_envelope_from_go(raw)

    raw = json.loads(_ZERO_LONGMEM_FIXTURE_PATH.read_text())
    evidence = raw["evidence"]
    assert isinstance(evidence, dict)
    providers = evidence["provider_evidence"]
    assert isinstance(providers, list)
    first = providers[0]
    assert isinstance(first, dict)
    first.update(
        {
            "requests": 1,
            "successes": 1,
            "receipted_requests": 1,
            "receipt_set_sha256": "e" * 64,
        }
    )
    raw["go_evidence_sha256"] = hashlib.sha256(
        json.dumps(evidence, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(ValidationError, match="cannot mix zero and positive"):
        longmem_envelope_from_go(raw)


def test_native_longmem_capability_order_is_frozen_before_normalization() -> None:
    fixture = copy.deepcopy(_fixture())
    longmem = fixture["longmemeval"]
    assert isinstance(longmem, dict)
    evidence = longmem["evidence"]
    assert isinstance(evidence, dict)
    score = evidence["score"]
    assert isinstance(score, dict)
    capabilities = score["per_capability"]
    assert isinstance(capabilities, list)
    score["per_capability"] = list(reversed(capabilities))
    longmem["go_evidence_sha256"] = hashlib.sha256(
        json.dumps(evidence, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()

    with pytest.raises(ConfirmationWireError, match="frozen order"):
        completion_report_from_go_fixture(
            fixture,
            ablation_coordinator_latency_ms=444,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("delta_micros", 499_999, "paired means"),
        ("semantic_factor_bps", 10_000, "semantic factor"),
        ("affected_call_count", 0, "affected-call count"),
    ],
)
def test_ablation_semantics_are_enforced_in_shared_model(
    field: str,
    value: int,
    message: str,
) -> None:
    evidence = _report().inference_ablation.evidence.model_dump(mode="json")
    evidence[field] = value
    with pytest.raises(ValidationError, match=message):
        AblationEvidence.model_validate(evidence)


def test_go_dimension_digest_is_verified_before_normalization() -> None:
    fixture = copy.deepcopy(_fixture())
    longmem = fixture["longmemeval"]
    assert isinstance(longmem, dict)
    evidence = longmem["evidence"]
    assert isinstance(evidence, dict)
    evidence["dataset_revision"] = "fabricated"

    with pytest.raises(ConfirmationWireError, match="Go evidence digest mismatch"):
        completion_report_from_go_fixture(
            fixture,
            ablation_coordinator_latency_ms=444,
        )


def test_shared_go_adapter_rejects_self_consistent_v1_positive_claim() -> None:
    fixture = copy.deepcopy(_fixture())
    dimension = fixture["inference_ablation"]
    assert isinstance(dimension, dict)
    evidence = dimension["evidence"]
    assert isinstance(evidence, dict)
    evidence.update(
        {
            "status": "passed",
            "reason": "threshold_met",
            "ablated_mean": 0.7,
            "delta": 0.2,
            "semantic_factor": 1,
        }
    )
    dimension["go_evidence_sha256"] = hashlib.sha256(
        json.dumps(evidence, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()

    with pytest.raises(ConfirmationWireError, match="unsupported Go ablation"):
        completion_report_from_go_fixture(
            fixture,
            ablation_coordinator_latency_ms=444,
        )


@pytest.mark.parametrize("mode", ["shadow", "enforce"])
def test_active_counterfactual_unavailability_has_no_numeric_gate(mode: str) -> None:
    evidence = _report().inference_ablation.evidence.model_dump(mode="json")
    evidence.update(
        {
            "mode": mode,
            "status": "unavailable",
            "reason": "counterfactual_proof_unavailable",
            "baseline_scores_sha256": None,
            "ablated_scores_sha256": None,
            "baseline_mean_micros": None,
            "ablated_mean_micros": None,
            "delta_micros": None,
            "semantic_factor_bps": None,
            "applied_factor_bps": None,
        }
    )

    parsed = AblationEvidence.model_validate(evidence)
    assert parsed.status == "unavailable"
    assert parsed.delta_micros is None


def test_counterfactual_reason_cannot_label_a_completed_result() -> None:
    evidence = _report().inference_ablation.evidence.model_dump(mode="json")
    evidence["reason"] = "counterfactual_proof_unavailable"

    with pytest.raises(ValidationError, match="active unavailable"):
        AblationEvidence.model_validate(evidence)
