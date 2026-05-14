from fed_adapter.data.splits import build_stratified_clients, wizard_stratification_labels


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

    from fed_adapter.data.splits import SplitRequest, create_split

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