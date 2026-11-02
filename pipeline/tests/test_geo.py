import pytest

from pipeline.geo import chip_corners_to_lonlat, parse_oam_name, tile_bounds_3857


def test_parse_oam_name():
    assert parse_oam_name("OAM-100-200-18.png") == (100, 200, 18)
    assert parse_oam_name("OAM-100-200-18") == (100, 200, 18)
    assert parse_oam_name("random.png") is None


def test_tile_bounds_3857_whole_world_at_z0():
    left, bottom, right, top = tile_bounds_3857(0, 0, 0)
    assert left == pytest.approx(-20037508.34, abs=1.0)
    assert right == pytest.approx(20037508.34, abs=1.0)
    assert bottom == pytest.approx(-20037508.34, abs=1.0)
    assert top == pytest.approx(20037508.34, abs=1.0)


def test_chip_corners_to_lonlat_top_left_pixel_is_tile_top_left():
    # tile (0, 0, 1) is the NW quadrant of the world; pixel (0, 0) is its top-left corner
    (lon, lat), = chip_corners_to_lonlat([(0, 0)], 0, 0, 1, 256, 256)
    assert lon == pytest.approx(-180.0, abs=0.01)
    assert lat == pytest.approx(85.0511, abs=0.1)


def test_chip_corners_to_lonlat_center_pixel_is_tile_center():
    (lon, lat), = chip_corners_to_lonlat([(128, 128)], 0, 0, 1, 256, 256)
    assert lon == pytest.approx(-90.0, abs=0.01)
