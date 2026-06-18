"""Unit tests for the Stage-1/Stage-2 PDI primitives (docs/adr/0005).

These pin each named operation on its own; the composed stage behaviour
(detect_clusters / refine_to_name_label) is pinned in test_run.py.
"""

import numpy as np

import pdiseg


def _draw_text_block(img, x0, y0, x1, y1, bg=30, ink=220):
    img[y0:y1, x0:x1] = bg
    for x in range(x0, x1, 8):
        img[y0:y1, x : x + 3] = ink


# --- Stage 1 -----------------------------------------------------------------


def test_text_mask_is_empty_on_a_uniform_image():
    img = np.full((120, 120), 120, dtype=np.uint8)
    assert not pdiseg.text_mask(img).any()


def test_text_mask_fires_on_locally_bright_strokes():
    img = np.full((120, 200), 30, dtype=np.uint8)
    _draw_text_block(img, 40, 40, 160, 90)

    mask = pdiseg.text_mask(img)

    assert mask.shape == img.shape
    assert mask.dtype == np.bool_
    # The strokes are inside the block; the flat surround stays quiet.
    assert mask[40:90, 40:160].any()
    assert not mask[:30, :30].any()


def test_close_mask_merges_strokes_into_a_solid_block():
    img = np.full((120, 200), 30, dtype=np.uint8)
    _draw_text_block(img, 40, 40, 160, 90)
    mask = pdiseg.text_mask(img)

    closed = pdiseg.close_mask(mask)

    # Closing fills the inter-stroke gaps, so the block is denser than raw strokes.
    assert closed[40:90, 40:160].sum() > mask[40:90, 40:160].sum()


def test_label_components_counts_separate_blobs():
    mask = np.zeros((100, 200), dtype=bool)
    mask[20:40, 20:40] = True
    mask[20:40, 120:140] = True

    labels = pdiseg.label_components(mask)

    assert labels.max() == 2
    assert pdiseg.label_components(np.zeros((10, 10), dtype=bool)).max() == 0


def test_boxes_from_components_drops_below_min_area():
    mask = np.zeros((100, 200), dtype=bool)
    mask[10:60, 10:110] = True  # big: 50*100 = 5000 px
    mask[80:85, 180:185] = True  # speck: 25 px
    labels = pdiseg.label_components(mask)

    boxes = pdiseg.boxes_from_components(labels, min_area=800)

    assert len(boxes) == 1
    x, y, w, h = boxes[0]
    assert (w, h) == (100, 50) and (x, y) == (10, 10)


def test_detect_clusters_equals_its_primitive_composition():
    img = np.full((200, 400), 30, dtype=np.uint8)
    _draw_text_block(img, 150, 80, 300, 140)

    composed = pdiseg.boxes_from_components(
        pdiseg.label_components(pdiseg.close_mask(pdiseg.text_mask(img)))
    )
    assert pdiseg.detect_clusters(img) == composed


# --- Stage 2 -----------------------------------------------------------------


def test_otsu_dark_mask_is_none_on_a_uniform_region():
    assert pdiseg.otsu_dark_mask(np.full((40, 40), 100, dtype=np.uint8)) is None
    assert pdiseg.otsu_dark_mask(np.empty((0, 0), dtype=np.uint8)) is None


def test_otsu_dark_mask_marks_the_dark_half():
    region = np.empty((40, 80), dtype=np.uint8)
    region[:, :40] = 210
    region[:, 40:] = 40

    mask = pdiseg.otsu_dark_mask(region)

    assert mask is not None
    assert mask[:, 40:].all()  # dark half is True
    assert not mask[:, :40].any()  # bright half is False


def test_largest_component_box_picks_the_bigger_blob():
    mask = np.zeros((100, 200), dtype=bool)
    mask[10:60, 10:90] = True  # big
    mask[80:90, 180:190] = True  # small

    box = pdiseg.largest_component_box(mask)

    assert box == (10, 10, 80, 50)


def test_largest_component_box_is_none_on_an_empty_mask():
    assert pdiseg.largest_component_box(np.zeros((20, 20), dtype=bool)) is None


# --- Stage 1 dark-badge variant (ADR 0006) -----------------------------------


def _draw_dark_badge(img, x0, y0, x1, y1, badge=40, ink=220):
    """A dark rounded-rectangle badge carrying bright text strokes."""
    img[y0:y1, x0:x1] = badge
    for x in range(x0 + 4, x1 - 2, 8):
        img[y0 + 4 : y1 - 4, x : x + 3] = ink


def test_dark_mask_selects_the_dark_badge_on_a_bright_frame():
    img = np.full((300, 500), 170, dtype=np.uint8)  # bright (glare-like) surround
    _draw_dark_badge(img, 150, 100, 330, 180)

    mask = pdiseg.dark_mask(img, percentile=20)

    assert mask.dtype == np.bool_
    assert mask[100:180, 150:330].any()  # badge background fires
    assert not mask[:80, :80].any()  # bright surround stays quiet


def test_dark_mask_is_empty_on_a_uniform_image():
    assert not pdiseg.dark_mask(np.full((50, 50), 120, dtype=np.uint8)).any()


def test_open_mask_drops_isolated_specks():
    mask = np.zeros((60, 60), dtype=bool)
    mask[10:40, 10:40] = True  # solid block survives
    mask[50, 50] = True  # lone speck removed

    opened = pdiseg.open_mask(mask)

    assert opened[10:40, 10:40].any()
    assert not opened[50, 50]


def test_detect_dark_badges_finds_a_single_badge():
    img = np.full((300, 500), 170, dtype=np.uint8)
    _draw_dark_badge(img, 150, 100, 330, 180)

    boxes = pdiseg.detect_dark_badges(img)

    assert len(boxes) == 1
    x, y, w, h = boxes[0]
    cx, cy = x + w / 2, y + h / 2
    assert 150 <= cx <= 330 and 100 <= cy <= 180


def test_detect_dark_badges_separates_two_distant_badges():
    img = np.full((300, 600), 170, dtype=np.uint8)
    _draw_dark_badge(img, 40, 100, 200, 180)
    _draw_dark_badge(img, 380, 100, 540, 180)

    assert len(pdiseg.detect_dark_badges(img)) == 2


def test_detect_dark_badges_is_empty_on_a_uniform_image():
    assert pdiseg.detect_dark_badges(np.full((300, 400), 120, dtype=np.uint8)) == []


# --- Stage 1 dark-relief variant (ADR 0007) ----------------------------------


def test_dark_relief_is_zero_on_a_uniform_image():
    # Closing of a flat field equals the field, so the top-hat is identically zero.
    assert not pdiseg.dark_relief(np.full((80, 80), 120, dtype=np.uint8)).any()


def test_dark_relief_lights_up_a_locally_dark_badge():
    img = np.full((300, 500), 170, dtype=np.uint8)
    _draw_dark_badge(img, 150, 100, 330, 180, badge=110)  # mid-grey, not globally dark

    relief = pdiseg.dark_relief(img, size=51)

    assert relief.dtype == np.uint8
    assert relief[100:180, 150:330].max() > 0  # the badge is darker than its surround
    assert relief[:80, :80].max() == 0  # the flat surround stays at zero relief


def test_relief_mask_is_empty_on_a_uniform_image():
    assert not pdiseg.relief_mask(pdiseg.dark_relief(np.full((80, 80), 90, dtype=np.uint8))).any()


def test_dark_relief_catches_a_grey_badge_that_global_dark_mask_misses():
    """The failure ADR 0007 fixes: a mid-grey badge missed by the global percentile.

    A large darker region elsewhere eats the darkest-20% budget, so ``dark_mask``
    never reaches the grey badge. The local top-hat ignores the large region (bigger
    than its structuring element) and surfaces the badge by local contrast.
    """
    img = np.full((300, 600), 180, dtype=np.uint8)
    img[:, :210] = np.linspace(0, 70, 210, dtype=np.uint8)  # big dark gradient, ~35% of frame
    _draw_dark_badge(img, 360, 110, 510, 190, badge=120)  # grey badge on the bright side
    badge = (slice(110, 190), slice(360, 510))

    missed = pdiseg.dark_mask(img, percentile=20)
    assert not missed[badge].any()  # global percentile never reaches the grey badge

    caught = pdiseg.relief_mask(pdiseg.dark_relief(img, size=51), percentile=20)
    assert caught[badge].any()  # local relief surfaces it anyway


def test_detect_dark_relief_badges_finds_a_grey_badge():
    img = np.full((300, 500), 170, dtype=np.uint8)
    _draw_dark_badge(img, 150, 100, 330, 180, badge=110)

    boxes = pdiseg.detect_dark_relief_badges(img)

    assert len(boxes) == 1
    x, y, w, h = boxes[0]
    cx, cy = x + w / 2, y + h / 2
    assert 150 <= cx <= 330 and 100 <= cy <= 180


def test_detect_dark_relief_badges_is_empty_on_a_uniform_image():
    assert pdiseg.detect_dark_relief_badges(np.full((300, 400), 120, dtype=np.uint8)) == []
