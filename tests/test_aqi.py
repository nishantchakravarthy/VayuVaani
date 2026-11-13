import numpy as np

from vayuvaani import aqi


def test_pm25_breakpoints():
    assert aqi.pm25_sub_index(0) == 0
    assert aqi.pm25_sub_index(30) == 50
    assert aqi.pm25_sub_index(60) == 100
    assert aqi.pm25_sub_index(90) == 200
    assert aqi.pm25_sub_index(120) == 300
    assert aqi.pm25_sub_index(250) == 400
    assert aqi.pm25_sub_index(1000) == 500


def test_categories():
    assert aqi.pm25_category(25) == "Good"
    assert aqi.pm25_category(45) == "Satisfactory"
    assert aqi.pm25_category(75) == "Moderate"
    assert aqi.pm25_category(100) == "Poor"
    assert aqi.pm25_category(200) == "Very Poor"
    assert aqi.pm25_category(300) == "Severe"


def test_missing_values():
    out = aqi.pm25_sub_index(np.array([10.0, np.nan]))
    assert out[0] > 0 and np.isnan(out[1])
    assert aqi.category(np.nan) is None


def test_threshold():
    assert aqi.pm25_threshold("Poor") == 90
