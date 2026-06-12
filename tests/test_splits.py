from fed_adapter.data.splits import (
    SplitRequest,
    build_stratified_clients,
    create_split,
    wizard_stratification_labels,
)


def test_build_stratified_clients_preserves_quotas():
    records = [
        {"instruction": "a", "category": "x"},
        {"instruction": "b", "category": "x"},
        {"instruction": "c", "category": "y"},
        {"instruction": "d", "category": "y"},
    ]
    labels = [record["category"] for record in records]

    clients, counts = build_stratified_clients(records, quotas=[1, 3], seed=1, labels=labels)

    assert [len(client) for client in clients] == [1, 3]
    assert sum(sum(client_counts.values()) for client_counts in counts.values()) == 4


def test_wizard_stratification_labels_include_family_and_bucket():
    labels, edges = wizard_stratification_labels(
        [
            {"instruction": "Write python code"},
            {"instruction": "Summarize this article"},
            {"instruction": "What is gravity"},
        ],
        num_length_buckets=2,
    )

    assert len(labels) == 3
    assert any(label.startswith("code__") for label in labels)
    assert any(label.startswith("analysis_classification__") for label in labels)
    assert all("__instruction_len_q" in label for label in labels)
    assert all(isinstance(edge, int) for edge in edges)



def test_missing_source_split_error_explains_relative_paths(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError) as exc_info:
        create_split(
            SplitRequest(
                dataset="wizard",
                mode="stratified_keep_sizes",
                num_clients=2,
                source_root=tmp_path / "missing_data_wiz",
                output_root=tmp_path / "data_wiz_stratified",
            )
        )

    message = str(exc_info.value)
    assert "Could not find the source federated split files" in message
    assert "Current working directory" in message
    assert "absolute --source-root" in message
    assert "local_training_0.json" in message


def test_glue_source_split_rewrites_mnli_with_mismatched_validation(tmp_path):
    import json

    source = tmp_path / "source_mnli"
    source.mkdir()
    (source / "local_training_0.json").write_text(
        json.dumps(
            [
                {"premise": "p0", "hypothesis": "h0", "label": 0},
                {"premise": "p1", "hypothesis": "h1", "label": 1},
                {"premise": "p2", "hypothesis": "h2", "label": 2},
            ]
        )
    )
    (source / "local_training_1.json").write_text(
        json.dumps(
            [
                {"premise": "p3", "hypothesis": "h3", "label": 0},
                {"premise": "p4", "hypothesis": "h4", "label": 1},
                {"premise": "p5", "hypothesis": "h5", "label": 2},
            ]
        )
    )
    (source / "global_val.json").write_text(
        json.dumps([{"premise": "pm", "hypothesis": "hm", "label": 1}])
    )
    (source / "global_val_mismatched.json").write_text(
        json.dumps([{"premise": "px", "hypothesis": "hx", "label": 2}])
    )

    output_dir = create_split(
        SplitRequest(
            dataset="glue",
            mode="stratified",
            task_name="mnli",
            num_clients=2,
            source_split_dir=source,
            output_root=tmp_path / "data_mnli_stratified",
            seed=0,
        )
    )

    assert output_dir == tmp_path / "data_mnli_stratified" / "2"
    assert (output_dir / "global_val.json").exists()
    assert (output_dir / "global_val_mismatched.json").exists()
    assert len(json.loads((output_dir / "local_training_0.json").read_text())) == 3
    assert len(json.loads((output_dir / "local_training_1.json").read_text())) == 3
    metadata = json.loads((output_dir / "split_metadata.json").read_text())
    assert metadata["task_name"] == "mnli"
    assert metadata["extra_validation_sizes"] == {"global_val_mismatched.json": 1}


def test_glue_stsb_stratification_uses_score_buckets(tmp_path):
    import json

    source = tmp_path / "source_stsb"
    source.mkdir()
    records = [
        {"sentence1": f"a{i}", "sentence2": f"b{i}", "label": float(i)}
        for i in range(6)
    ]
    (source / "local_training_0.json").write_text(json.dumps(records[:3]))
    (source / "local_training_1.json").write_text(json.dumps(records[3:]))
    (source / "global_val.json").write_text(json.dumps(records[:2]))

    output_dir = create_split(
        SplitRequest(
            dataset="glue",
            mode="stratified",
            task_name="stsb",
            num_clients=2,
            source_split_dir=source,
            output_root=tmp_path / "data_stsb_stratified",
            stsb_num_label_buckets=3,
            seed=1,
        )
    )

    metadata = json.loads((output_dir / "split_metadata.json").read_text())
    assert sorted(metadata["train_label_counts"]) == ["score_q0", "score_q1", "score_q2"]
    assert all(
        label in {"score_q0", "score_q1", "score_q2"}
        for counts in metadata["client_label_counts"].values()
        for label in counts
    )
