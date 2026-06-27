import pdiseg


def test_reload_pipeline_modules_refreshes_masks_schema() -> None:
    pdiseg.reload_pipeline_modules()
    pdiseg.assert_pipeline_schema()
    from pdiseg.detection.masks import CandidateMasks

    assert "dog_text" in CandidateMasks.__dataclass_fields__
    assert "edge_density" in CandidateMasks.__dataclass_fields__
