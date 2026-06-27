import imageio.v3 as iio
import numpy as np

import pdiseg


def _draw_text_block(img, x0, y0, x1, y1, bg=30, ink=220):
    img[y0:y1, x0:x1] = bg
    for x in range(x0, x1, 8):
        img[y0:y1, x : x + 3] = ink


def _gradient_frame(h, w):
    return np.tile(np.linspace(0, 255, w).astype(np.uint8), (h, 1))


def test_select_sample_images_picks_one_per_class(tmp_path):
    for name in ("ClassA", "ClassB"):
        folder = tmp_path / name
        folder.mkdir()
        for i in range(3):
            iio.imwrite(folder / f"img{i}.png", np.full((40, 40), 120, dtype=np.uint8))

    picked = pdiseg.select_sample_images(tmp_path, per_class=1)
    assert len(picked) == 2
    assert {p.parent.name for p in picked} == {"ClassA", "ClassB"}
    assert all(p.name == "img0.png" for p in picked)


def test_run_debug_sample_writes_crops_and_bundles(tmp_path):
    dataset = tmp_path / "dataset" / "Peito"
    dataset.mkdir(parents=True)
    frame = _gradient_frame(300, 500)
    _draw_text_block(frame, 200, 100, 360, 180)
    iio.imwrite(dataset / "sample.png", frame)

    result = tmp_path / "debug_result" / "result"
    bundles = tmp_path / "debug_result" / "bundles"
    report = pdiseg.run_debug_sample(dataset.parent, result, bundle_root=bundles, per_class=1)

    assert report.dataset_report.total_frames == 1
    assert report.dataset_report.total_crops >= 1
    assert (result / "Peito" / "sample_segmented_1.png").exists()
    assert (bundles / "Peito" / "sample_preprocess.png").exists()
    assert (bundles / "Peito" / "sample_pipeline.png").exists()
    assert report.frames[0].class_name == "Peito"
    assert report.frames[0].snapshot.detection.labels == report.frames[0].labels


def test_build_sample_views_recovers_missing_snapshot(tmp_path):
    dataset = tmp_path / "dataset" / "Peito"
    dataset.mkdir(parents=True)
    frame = _gradient_frame(300, 500)
    _draw_text_block(frame, 200, 100, 360, 180)
    iio.imwrite(dataset / "sample.png", frame)

    report = pdiseg.run_debug_sample(dataset.parent, tmp_path / "out", per_class=1)
    item = report.frames[0]
    stale = pdiseg.DebugFrameResult(
        class_name=item.class_name,
        source=item.source,
        labels=item.labels,
        bundle_dir=item.bundle_dir,
    )
    views = pdiseg.build_sample_views(
        pdiseg.DebugSampleReport(
            dataset_report=report.dataset_report,
            frames=[stale],
            result_root=report.result_root,
            bundle_root=report.bundle_root,
        )
    )
    assert len(views) == 1
    assert len(views[0].snapshot.detection.labels) == len(item.labels)


def test_plot_helpers_render_without_display(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset = tmp_path / "dataset" / "Peito"
    dataset.mkdir(parents=True)
    frame = _gradient_frame(300, 500)
    _draw_text_block(frame, 200, 100, 360, 180)
    iio.imwrite(dataset / "sample.png", frame)

    report = pdiseg.run_debug_sample(dataset.parent, tmp_path / "out", per_class=1)
    item = report.frames[0]
    image = pdiseg.load_image(item.source)
    title = frame_title = f"{item.class_name}\n{item.source.name}"
    items = [(frame_title, item.snapshot)]

    pdiseg.plot_all_preprocessed(items)
    pdiseg.plot_all_masks(items)
    pdiseg.plot_all_overlays([image], items)
    pdiseg.plot_all_preprocess_stages(items)
    pdiseg.plot_all_mask_layers(items)
    pdiseg.plot_all_detection_stage([image], items, stage="candidates")
    pdiseg.plot_frame_pipeline(image, item.snapshot, title=title)
    pdiseg.plot_frame_full_debug(image, item.snapshot, title=title)
    pdiseg.plot_all_crops([image], [item.labels], [frame_title])
    pdiseg.plot_bundle_gallery(item.bundle_dir, item.source.stem)
    plt.close("all")
