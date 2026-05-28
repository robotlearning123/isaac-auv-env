import json
from pathlib import Path

from oceanscale.cli import (
    _build_parser,
    _eval_robustness_profile,
    _eval_robustness_report,
    _eval_robustness_run,
    _eval_robustness_suite,
)
from oceanscale.evaluation import (
    ACTION_MODES,
    EvaluationAxis,
    EvaluationCase,
    EvaluationProfile,
    EvaluationRecord,
    _action_from_observation,
    collect_run_metadata,
    evaluation_records_from_jsonl,
    evaluation_report_to_markdown,
    generate_evaluation_cases,
    profile_to_dict,
    profile_to_markdown,
    records_to_jsonl,
    resolve_case_evaluator,
    resolve_zero_action_evaluator,
    run_evaluation_profile,
    summarize_evaluation_records,
    summarize_evaluation_run,
    underwater_robustness_profile,
)


def test_underwater_robustness_profile_has_underwater_axes() -> None:
    profile = underwater_robustness_profile(episodes_per_case=7, seed=123)

    assert profile.name == "underwater_robustness_v0"
    assert profile.episodes_per_case == 7
    assert profile.seed == 123
    assert {axis.category for axis in profile.axes} == {"ocean", "vehicle", "sensor", "task"}
    assert "success_rate" in profile.metrics
    assert any(axis.parameter == "current_speed_mps" for axis in profile.axes)
    assert any(axis.parameter == "visibility_m" for axis in profile.axes)


def test_generate_evaluation_cases_is_deterministic_and_includes_baseline() -> None:
    profile = underwater_robustness_profile(episodes_per_case=3, seed=42)

    cases = generate_evaluation_cases(profile)
    case_ids = [case.case_id for case in cases]

    assert case_ids[0] == "underwater_robustness_v0:baseline"
    assert case_ids == [case.case_id for case in generate_evaluation_cases(profile)]
    assert cases[0].overrides == {}
    assert cases[1].episodes == 3
    assert cases[1].seed == 43
    assert any(case.overrides.get("drag_scale") == 1.15 for case in cases)


def test_generate_evaluation_cases_slugs_negative_values() -> None:
    profile = EvaluationProfile(
        name="negative_value_probe",
        description="Probe case-id formatting for signed values.",
        task_family="test",
        episodes_per_case=1,
        seed=5,
        axes=(
            EvaluationAxis(
                category="ocean",
                name="signed current",
                parameter="current_offset_mps",
                unit="m/s",
                baseline=0.0,
                values=(-0.5,),
                rationale="Regression check for stable case IDs.",
            ),
        ),
        metrics=("success_rate",),
        claim_guardrails=("No measured claim.",),
    )

    cases = generate_evaluation_cases(profile)

    assert cases[1].case_id == "negative_value_probe:ocean:signed_current:m0p5"
    assert cases[1].overrides == {"current_offset_mps": -0.5}


def test_underwater_robustness_profile_rejects_zero_episodes() -> None:
    try:
        underwater_robustness_profile(episodes_per_case=0)
    except ValueError as exc:
        assert "episodes_per_case" in str(exc)
    else:
        raise AssertionError("episodes_per_case=0 should fail")


def test_profile_serializes_without_claiming_sim2real_results() -> None:
    profile = underwater_robustness_profile()

    data = profile_to_dict(profile)
    encoded = json.dumps(data)

    assert data["source_reference"]["adoption"] == "method_reference_only"
    assert "does not prove sim-to-real transfer" in " ".join(data["claim_guardrails"])
    assert "89%" not in encoded
    assert "foundation model" not in encoded.lower()


def test_profile_markdown_contains_protocol_tables() -> None:
    profile = underwater_robustness_profile(episodes_per_case=2)

    markdown = profile_to_markdown(profile)

    assert "# OceanScale Evaluation Profile: underwater_robustness_v0" in markdown
    assert "| Category | Axis | Parameter | Baseline | Values |" in markdown
    assert "Genesis World 1.0" in markdown
    assert "No sim-to-real claim" in markdown


def test_eval_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    parser = _build_parser()
    json_path = tmp_path / "reports" / "profile.json"
    md_path = tmp_path / "reports" / "profile.md"
    args = parser.parse_args(
        [
            "eval",
            "robustness-profile",
            "--episodes-per-case",
            "5",
            "--seed",
            "99",
            "--output-json",
            str(json_path),
            "--output-md",
            str(md_path),
        ]
    )

    _eval_robustness_profile(args)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = md_path.read_text(encoding="utf-8")
    assert data["episodes_per_case"] == 5
    assert data["seed"] == 99
    assert data["cases"][0]["case_id"] == "underwater_robustness_v0:baseline"
    assert "underwater_robustness_v0" in markdown


def test_eval_run_cli_writes_case_records_with_injected_evaluator(tmp_path: Path) -> None:
    parser = _build_parser()
    jsonl_path = tmp_path / "records.jsonl"
    summary_path = tmp_path / "summary.json"
    args = parser.parse_args(
        [
            "eval",
            "robustness-run",
            "--episodes-per-case",
            "2",
            "--case-limit",
            "2",
            "--action-mode",
            "proportional",
            "--output-jsonl",
            str(jsonl_path),
            "--summary-json",
            str(summary_path),
        ]
    )

    def fake_evaluator(case: EvaluationCase) -> dict[str, object]:
        return {
            "metrics": {
                "success_rate": 1.0,
                "episode_return": 1.0,
                "mean_position_error_m": 0.1,
                "max_position_error_m": 0.2,
                "energy_proxy": 0.0,
                "constraint_violations": 0,
                "wall_time_s": 0.01,
            },
            "applied_overrides": case.overrides,
            "unsupported_overrides": {},
            "environment_config": {
                "env_class": "FakeEnv",
                "task_name": "station-keeping",
                "action_mode": "proportional",
                "supported_overrides": sorted(case.overrides),
                "applied_overrides": case.overrides,
            },
        }

    _eval_robustness_run(args, evaluate_case=fake_evaluator)

    records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(records) == 2
    assert records[0]["task_name"] == "station-keeping"
    assert records[0]["action_mode"] == "proportional"
    assert records[0]["run_metadata"]["action_mode"] == "proportional"
    assert records[0]["run_metadata"]["task_name"] == "station-keeping"
    assert records[0]["run_metadata"]["profile_name"] == "underwater_robustness_v0"
    assert records[0]["run_metadata"]["profile_seed"] == 42
    assert records[0]["run_metadata"]["episodes_per_case"] == 2
    assert records[0]["run_metadata"]["case_limit"] == 2
    assert "machine" in records[0]["run_metadata"]
    assert "package_versions" in records[0]["run_metadata"]
    assert isinstance(records[0]["run_metadata"]["gpu_count"], int)
    assert isinstance(records[0]["run_metadata"]["gpu_inventory_known"], bool)
    assert records[0]["environment_config"]["env_class"] == "FakeEnv"
    assert records[0]["environment_config"]["task_name"] == "station-keeping"
    assert summary["environment_configs"][0]["env_class"] == "FakeEnv"
    assert "git_revision" in records[0]["run_metadata"]
    assert isinstance(records[0]["run_metadata"]["git_dirty"], bool)
    assert summary["case_count"] == 2
    assert summary["status_counts"] == {"completed": 2}
    assert summary["action_mode"] == "proportional"
    assert summary["run_metadata"] == records[0]["run_metadata"]


def test_evaluation_report_preserves_failed_cases_and_provenance() -> None:
    run_metadata = {
        "task_name": "station-keeping",
        "action_mode": "proportional",
        "device": "cpu",
        "steps_per_episode": 4,
        "git_revision": "abc123",
        "git_dirty": True,
        "python": "3.12.0",
        "platform": "test-platform",
        "package_versions": {"numpy": "2.4.4", "torch": "2.11.0+cu128"},
    }
    records = [
        EvaluationRecord(
            case_id="underwater_robustness_v0:baseline",
            task_name="station-keeping",
            action_mode="proportional",
            status="completed",
            episodes=2,
            seed=42,
            overrides={},
            applied_overrides={},
            unsupported_overrides={},
            metrics={
                "success_rate": 1.0,
                "mean_position_error_m": 0.2,
                "wall_time_s": 0.01,
            },
            wall_time_s=0.01,
            run_metadata=run_metadata,
            environment_config={
                "env_class": "CurrentStationKeepingEnv",
                "task_name": "station-keeping",
                "action_mode": "proportional",
                "device": "cpu",
                "supported_overrides": [
                    "current_direction_deg",
                    "current_speed_mps",
                    "target_offset_m",
                ],
            },
        ),
        EvaluationRecord(
            case_id="underwater_robustness_v0:vehicle:drag_scale:1p15",
            task_name="station-keeping",
            action_mode="proportional",
            status="failed",
            episodes=2,
            seed=43,
            overrides={"drag_scale": 1.15},
            applied_overrides={},
            unsupported_overrides={"drag_scale": 1.15},
            metrics={},
            wall_time_s=0.02,
            run_metadata=run_metadata,
            environment_config={
                "env_class": "CurrentStationKeepingEnv",
                "task_name": "station-keeping",
                "action_mode": "proportional",
                "device": "cpu",
                "supported_overrides": [
                    "current_direction_deg",
                    "current_speed_mps",
                    "target_offset_m",
                ],
            },
            error="RuntimeError: synthetic failure",
        ),
    ]

    loaded = evaluation_records_from_jsonl(records_to_jsonl(records))
    report = summarize_evaluation_records(loaded)
    markdown = evaluation_report_to_markdown(report)

    assert report["case_count"] == 2
    assert report["status_counts"] == {"completed": 1, "failed": 1}
    assert report["mean_metrics"]["success_rate"] == 1.0
    assert report["metric_distributions"]["success_rate"] == {
        "count": 1,
        "min": 1.0,
        "p50": 1.0,
        "p90": 1.0,
        "p95": 1.0,
        "max": 1.0,
    }
    assert report["unsupported_parameters"] == ["drag_scale"]
    assert report["failed_cases"] == [
        {
            "case_id": "underwater_robustness_v0:vehicle:drag_scale:1p15",
            "status": "failed",
            "error": "RuntimeError: synthetic failure",
        }
    ]
    assert report["run_metadata"] == run_metadata
    assert report["environment_configs"] == [
        {
            "env_class": "CurrentStationKeepingEnv",
            "task_name": "station-keeping",
            "action_mode": "proportional",
            "device": "cpu",
            "supported_overrides": [
                "current_direction_deg",
                "current_speed_mps",
                "target_offset_m",
            ],
        },
        {
            "env_class": "CurrentStationKeepingEnv",
            "task_name": "station-keeping",
            "action_mode": "proportional",
            "device": "cpu",
            "supported_overrides": [
                "current_direction_deg",
                "current_speed_mps",
                "target_offset_m",
            ],
            "requested_overrides": {"drag_scale": 1.15},
            "unsupported_overrides": {"drag_scale": 1.15},
        },
    ]
    assert "Failed Cases" in markdown
    assert "Metric Distributions" in markdown
    assert "Environment Configs" in markdown
    assert "env_class: CurrentStationKeepingEnv" in markdown
    assert "requested_overrides.drag_scale: 1.15" in markdown
    assert "unsupported_overrides.drag_scale: 1.15" in markdown
    assert "package_versions.numpy: 2.4.4" in markdown
    assert "{'numpy'" not in markdown
    assert "Claim Guardrails" in markdown
    assert "Failed and unsupported cases must remain visible" in markdown
    assert "Genesis World 1.0 evaluation-first pattern" in markdown
    assert "RuntimeError: synthetic failure" in markdown
    assert "not a sim-to-real result" in markdown


def test_evaluation_report_includes_tail_metric_distributions() -> None:
    records = [
        {
            "case_id": f"underwater_robustness_v0:case:{index}",
            "task_name": "station-keeping",
            "action_mode": "proportional",
            "status": "completed",
            "unsupported_overrides": {},
            "metrics": {"success_rate": success_rate, "mean_position_error_m": error},
            "run_metadata": {"task_name": "station-keeping"},
        }
        for index, (success_rate, error) in enumerate(
            [(1.0, 0.1), (0.0, 0.4), (0.5, 1.2), (1.0, 2.0)]
        )
    ]

    report = summarize_evaluation_records(records)

    assert report["mean_metrics"]["mean_position_error_m"] == 0.925
    assert report["metric_distributions"]["success_rate"] == {
        "count": 4,
        "min": 0.0,
        "p50": 1.0,
        "p90": 1.0,
        "p95": 1.0,
        "max": 1.0,
    }
    assert report["metric_distributions"]["mean_position_error_m"] == {
        "count": 4,
        "min": 0.1,
        "p50": 1.2,
        "p90": 2.0,
        "p95": 2.0,
        "max": 2.0,
    }


def test_eval_report_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    parser = _build_parser()
    jsonl_path = tmp_path / "records.jsonl"
    json_path = tmp_path / "reports" / "report.json"
    md_path = tmp_path / "reports" / "report.md"
    jsonl_path.write_text(
        json.dumps(
            {
                "case_id": "underwater_robustness_v0:baseline",
                "task_name": "waypoint",
                "action_mode": "zero",
                "status": "completed",
                "episodes": 1,
                "seed": 42,
                "overrides": {},
                "applied_overrides": {},
                "unsupported_overrides": {},
                "metrics": {"success_rate": 0.0, "wall_time_s": 0.01},
                "wall_time_s": 0.01,
                "environment_config": {
                    "env_class": "WaypointFollowingEnv",
                    "task_name": "waypoint",
                    "action_mode": "zero",
                    "n_waypoints": 1,
                },
                "run_metadata": {
                    "task_name": "waypoint",
                    "action_mode": "zero",
                    "device": "cpu",
                    "steps_per_episode": 1,
                    "git_revision": "abc123",
                    "git_dirty": False,
                    "python": "3.12.0",
                    "platform": "test-platform",
                },
                "error": None,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    args = parser.parse_args(
        [
            "eval",
            "robustness-report",
            "--input-jsonl",
            str(jsonl_path),
            "--output-json",
            str(json_path),
            "--output-md",
            str(md_path),
        ]
    )

    _eval_robustness_report(args)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = md_path.read_text(encoding="utf-8")
    assert data["case_count"] == 1
    assert data["task_name"] == "waypoint"
    assert data["action_mode"] == "zero"
    assert data["status_counts"] == {"completed": 1}
    assert data["environment_configs"][0]["env_class"] == "WaypointFollowingEnv"
    assert "OceanScale Evaluation Report" in markdown
    assert "Environment Configs" in markdown
    assert "waypoint" in markdown


def test_eval_suite_cli_writes_manifest_and_artifacts(tmp_path: Path) -> None:
    parser = _build_parser()
    output_dir = tmp_path / "suite"
    args = parser.parse_args(
        [
            "eval",
            "robustness-suite",
            "--tasks",
            "station-keeping",
            "waypoint",
            "--action-modes",
            "zero",
            "damped",
            "--episodes-per-case",
            "1",
            "--case-limit",
            "2",
            "--steps-per-episode",
            "1",
            "--device",
            "cpu",
            "--output-dir",
            str(output_dir),
        ]
    )

    def fake_factory(task_name: str, action_mode: str):
        def fake_evaluator(case: EvaluationCase) -> dict[str, object]:
            return {
                "metrics": {
                    "success_rate": 1.0,
                    "episode_return": 1.0,
                    "mean_position_error_m": 0.1,
                    "max_position_error_m": 0.2,
                    "energy_proxy": 0.0,
                    "constraint_violations": 0,
                    "wall_time_s": 0.01,
                },
                "applied_overrides": case.overrides,
                "unsupported_overrides": {},
                "environment_config": {
                    "env_class": "FakeSuiteEnv",
                    "task_name": task_name,
                    "action_mode": action_mode,
                    "supported_overrides": sorted(case.overrides),
                },
            }

        return fake_evaluator

    _eval_robustness_suite(args, evaluate_case_factory=fake_factory)

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["suite_type"] == "underwater_robustness_suite"
    assert manifest["profile"] == "underwater_robustness_v0"
    assert manifest["tasks"] == ["station-keeping", "waypoint"]
    assert manifest["action_modes"] == ["zero", "damped"]
    assert manifest["case_limit"] == 2
    assert manifest["steps_per_episode"] == 1
    assert manifest["source_reference"]["dependency"] == "none"
    assert len(manifest["artifacts"]) == 4

    for artifact in manifest["artifacts"]:
        assert artifact["case_count"] == 2
        assert artifact["status_counts"] == {"completed": 2}
        records_path = output_dir / artifact["records_jsonl"]
        summary_path = output_dir / artifact["summary_json"]
        report_json_path = output_dir / artifact["report_json"]
        report_md_path = output_dir / artifact["report_md"]
        assert records_path.exists()
        assert summary_path.exists()
        assert report_json_path.exists()
        assert report_md_path.exists()

        first_record = json.loads(records_path.read_text(encoding="utf-8").splitlines()[0])
        assert first_record["run_metadata"]["task_name"] == artifact["task_name"]
        assert first_record["run_metadata"]["action_mode"] == artifact["action_mode"]
        assert first_record["run_metadata"]["profile_name"] == "underwater_robustness_v0"
        assert first_record["environment_config"]["env_class"] == "FakeSuiteEnv"
        assert first_record["environment_config"]["task_name"] == artifact["task_name"]
        assert "OceanScale Evaluation Report" in report_md_path.read_text(encoding="utf-8")


def test_evaluation_report_rejects_empty_record_set() -> None:
    try:
        summarize_evaluation_records([])
    except ValueError as exc:
        assert "at least one record" in str(exc)
    else:
        raise AssertionError("empty record set should fail")


def test_resolve_zero_action_evaluator_supports_all_public_tasks() -> None:
    for task_name in ("station-keeping", "docking", "waypoint"):
        evaluator = resolve_zero_action_evaluator(
            task_name,
            steps_per_episode=1,
            device="cpu",
        )

        assert callable(evaluator)


def test_resolve_zero_action_evaluator_rejects_unknown_task() -> None:
    try:
        resolve_zero_action_evaluator("unknown", steps_per_episode=1, device="cpu")
    except ValueError as exc:
        assert "unknown task" in str(exc)
    else:
        raise AssertionError("unknown task should fail")


def test_resolve_case_evaluator_supports_zero_and_proportional_modes() -> None:
    for action_mode in ("zero", "proportional", "damped"):
        for task_name in ("station-keeping", "docking", "waypoint"):
            evaluator = resolve_case_evaluator(
                task_name,
                action_mode=action_mode,
                steps_per_episode=1,
                device="cpu",
            )

            assert callable(evaluator)


def test_action_modes_include_damped_mission_heuristic() -> None:
    assert ACTION_MODES == ("zero", "proportional", "damped")


def test_damped_action_uses_task_specific_velocity_terms() -> None:
    import numpy as np

    station_obs = np.zeros(29, dtype=np.float32)
    station_obs[:3] = [1.0, -2.0, 4.0]
    station_obs[7:10] = [0.5, -0.5, 2.0]

    docking_obs = np.zeros(31, dtype=np.float32)
    docking_obs[:3] = [1.0, -2.0, 4.0]
    docking_obs[10:13] = [0.5, -0.5, 2.0]

    station_action = _action_from_observation(
        station_obs,
        "damped",
        np,
        task_name="station-keeping",
    )
    docking_action = _action_from_observation(
        docking_obs,
        "damped",
        np,
        task_name="docking",
    )

    assert np.allclose(station_action[:3], [0.375, -0.825, 1.0])
    assert np.allclose(docking_action[:3], [0.375, -0.825, 1.0])


def test_resolve_case_evaluator_rejects_unknown_action_mode() -> None:
    try:
        resolve_case_evaluator(
            "station-keeping",
            action_mode="trained-policy",
            steps_per_episode=1,
            device="cpu",
        )
    except ValueError as exc:
        assert "unknown action mode" in str(exc)
    else:
        raise AssertionError("unknown action mode should fail")


def test_run_evaluation_profile_collects_case_records() -> None:
    profile = underwater_robustness_profile(episodes_per_case=2, seed=11)
    run_metadata = {
        "task_name": "fake_task",
        "action_mode": "zero",
        "device": "cpu",
        "steps_per_episode": 1,
        "git_revision": "abc123",
        "git_dirty": False,
        "python": "3.12.0",
        "platform": "test-platform",
    }

    def fake_evaluator(case: EvaluationCase) -> dict[str, object]:
        return {
            "metrics": {
                "success_rate": 1.0 if case.category == "baseline" else 0.5,
                "episode_return": 2.0,
                "mean_position_error_m": 0.25,
                "max_position_error_m": 0.5,
                "energy_proxy": 0.0,
                "constraint_violations": 0,
                "wall_time_s": 0.01,
            },
            "applied_overrides": case.overrides,
            "unsupported_overrides": {},
            "environment_config": {
                "env_class": "FakeProfileEnv",
                "task_name": "fake_task",
                "action_mode": "zero",
                "supported_overrides": sorted(case.overrides),
            },
        }

    records = run_evaluation_profile(
        profile,
        task_name="fake_task",
        evaluate_case=fake_evaluator,
        case_limit=3,
        action_mode="zero",
        run_metadata=run_metadata,
    )
    summary = summarize_evaluation_run(
        profile,
        records,
        task_name="fake_task",
        action_mode="zero",
        run_metadata=run_metadata,
    )
    jsonl = records_to_jsonl(records)

    assert [record.case_id for record in records] == [
        "underwater_robustness_v0:baseline",
        "underwater_robustness_v0:ocean:current_strength:0p0",
        "underwater_robustness_v0:ocean:current_strength:0p5",
    ]
    assert records[0].action_mode == "zero"
    assert records[0].run_metadata == run_metadata
    assert records[0].environment_config["env_class"] == "FakeProfileEnv"
    assert records[1].applied_overrides == {"current_speed_mps": 0.0}
    assert records[1].environment_config["requested_overrides"] == {"current_speed_mps": 0.0}
    assert records[1].environment_config["applied_overrides"] == {"current_speed_mps": 0.0}
    assert summary["case_count"] == 3
    assert summary["status_counts"] == {"completed": 3}
    assert summary["mean_metrics"]["success_rate"] == 2.0 / 3.0
    assert summary["metric_distributions"]["success_rate"] == {
        "count": 3,
        "min": 0.5,
        "p50": 0.5,
        "p90": 1.0,
        "p95": 1.0,
        "max": 1.0,
    }
    assert summary["action_mode"] == "zero"
    assert summary["run_metadata"] == run_metadata
    assert summary["environment_configs"][0]["env_class"] == "FakeProfileEnv"
    assert jsonl.count("\n") == 3


def test_records_to_jsonl_converts_numpy_scalars_in_metrics_and_environment_config() -> None:
    import numpy as np

    record = EvaluationRecord(
        case_id="underwater_robustness_v0:baseline",
        task_name="station-keeping",
        action_mode="zero",
        status="completed",
        episodes=1,
        seed=42,
        overrides={},
        applied_overrides={},
        unsupported_overrides={},
        metrics={"success_rate": np.float32(1.0), "constraint_violations": np.int64(0)},
        wall_time_s=0.01,
        run_metadata={"gpu_count": np.int64(1)},
        environment_config={
            "env_class": "FakeEnv",
            "observation_dim": np.int64(29),
            "nested": {"gain": np.float32(0.5)},
        },
    )

    loaded = json.loads(records_to_jsonl([record]).splitlines()[0])

    assert loaded["metrics"] == {"constraint_violations": 0.0, "success_rate": 1.0}
    assert loaded["run_metadata"]["gpu_count"] == 1
    assert loaded["environment_config"]["observation_dim"] == 29
    assert loaded["environment_config"]["nested"]["gain"] == 0.5


def test_collect_run_metadata_is_json_safe() -> None:
    metadata = collect_run_metadata(
        task_name="station-keeping",
        action_mode="zero",
        device="cpu",
        steps_per_episode=3,
        profile_name="underwater_robustness_v0",
        profile_seed=42,
        episodes_per_case=2,
        case_limit=5,
    )
    encoded = json.dumps(metadata)

    assert metadata["task_name"] == "station-keeping"
    assert metadata["action_mode"] == "zero"
    assert metadata["device"] == "cpu"
    assert metadata["steps_per_episode"] == 3
    assert metadata["profile_name"] == "underwater_robustness_v0"
    assert metadata["profile_seed"] == 42
    assert metadata["episodes_per_case"] == 2
    assert metadata["case_limit"] == 5
    assert "git_revision" in metadata
    assert isinstance(metadata["git_dirty"], bool)
    assert "python" in metadata
    assert "platform" in metadata
    assert "machine" in metadata
    assert "processor" in metadata
    assert isinstance(metadata["gpu_inventory_known"], bool)
    assert isinstance(metadata["gpu_count"], int)
    assert "gpu_names" in metadata
    assert "nvidia_driver_version" in metadata
    assert isinstance(metadata["package_versions"], dict)
    package_versions = metadata["package_versions"]
    assert "numpy" in package_versions
    assert "torch" in package_versions
    assert "warp" in package_versions
    assert encoded
