import pandas as pd
from unittest.mock import patch, MagicMock
from src.module_1.module_1_meteo_api import (
    compute_monthly_statistics,
    _request_with_cooloff,
    get_data_meteo_api,
    VARIABLES,
)


def test_compute_monthly_statistics():
    fake_data = {
        "time": ["2020-01-01", "2020-01-15", "2020-02-10"],
        "city": ["Madrid", "Madrid", "Madrid"],
        "temperature_2m_mean": [10.0, 12.0, 15.0],
        "precipitation_sum": [0.0, 5.0, 0.0],
        "wind_speed_10m_max": [10.0, 15.0, 20.0],
    }
    df = pd.DataFrame(fake_data)

    result = compute_monthly_statistics(df, VARIABLES)

    assert len(result) == 2
    assert result.iloc[0]["temperature_2m_mean_mean"] == 11.0
    assert result.iloc[0]["precipitation_sum_max"] == 5.0


def test_request_with_cooloff_success():
    # Mock a successful API response without hitting the network
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.raise_for_status = MagicMock()

    with patch(
        "src.module_1.module_1_meteo_api.requests.get",
        return_value=fake_response,
    ) as mock_get:
        result = _request_with_cooloff(
            "http://fake-url.com", headers={}, num_attempts=3
        )

    assert result is fake_response
    mock_get.assert_called_once()


def test_get_data_meteo_api_builds_correct_url():
    # Mock request_wrapper to verify URL construction with correct params
    fake_payload = {
        "daily": {
            "time": ["2020-01-01"],
            "temperature_2m_mean": [10.0],
            "precipitation_sum": [0.0],
            "wind_speed_10m_max": [5.0],
        }
    }

    with patch(
        "src.module_1.module_1_meteo_api.request_wrapper",
        return_value=fake_payload,
    ) as mock_wrapper:
        result = get_data_meteo_api(
            longitude=-3.7,
            latitude=40.4,
            start_date="2020-01-01",
            end_date="2020-01-31",
        )

    assert result == fake_payload
    mock_wrapper.assert_called_once()

    called_url = mock_wrapper.call_args[0][0]
    assert "latitude=40.4" in called_url
    assert "longitude=-3.7" in called_url
    assert "start_date=2020-01-01" in called_url