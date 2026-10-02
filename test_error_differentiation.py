"""
Tests for distinguishing between rate limit (429) and temporary unavailable (503/404),
immediate key rotation on 429, and fast failure on unavailable models.
"""
import pytest
from unittest.mock import MagicMock, patch

from models.base import ModelUnavailableError, RateLimitExhaustedError, QuotaExhaustedError
from models.gemini_model import GeminiModel, GeminiKeyManager, parse_gemini_rate_limit
from models.groq_model import GroqModel, GroqKeyManager
from pipeline.premise_architect import PremiseArchitect
from app import _handle_llm_exception, app


def test_parse_gemini_rate_limit_daily():
    err_msg = (
        "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota. "
        "Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, "
        "limit: 20, model: gemini-3.5-flash\\nPlease retry in 5h41m8.85832575s.', 'details': [{'@type': 'RetryInfo', 'retryDelay': '20468s'}]}}"
    )
    wait_sec, is_daily = parse_gemini_rate_limit(err_msg)
    assert is_daily is True
    assert 20400 <= wait_sec <= 20500


def test_parse_gemini_rate_limit_per_minute():
    err_msg = "429 RESOURCE_EXHAUSTED. Resource has been exhausted (e.g. check quota). Please retry after 15s."
    wait_sec, is_daily = parse_gemini_rate_limit(err_msg)
    assert is_daily is False
    assert wait_sec == 15.0


def test_gemini_503_raises_model_unavailable_immediately_no_retry():
    model = GeminiModel(model="gemini-3.6-flash")
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError(
        "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand. "
        "Spikes in demand are usually temporary. Please try again later.', 'status': 'UNAVAILABLE'}}"
    )
    with patch.object(GeminiModel, "client", mock_client):
        with pytest.raises(ModelUnavailableError) as exc_info:
            model.generate("test prompt")
        
        err = exc_info.value
        assert err.is_temporary is True
        assert err.status_code == 503
        assert "503 UNAVAILABLE" in str(err)
        assert "temporary server traffic spike, NOT a rate limit" in str(err)
        assert mock_client.models.generate_content.call_count == 1


def test_gemini_404_raises_model_unavailable_not_found():
    model = GeminiModel(model="gemini-2.5-flash")
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError(
        "404 NOT_FOUND. {'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available.', 'status': 'NOT_FOUND'}}"
    )
    with patch.object(GeminiModel, "client", mock_client):
        with pytest.raises(ModelUnavailableError) as exc_info:
            model.generate("test prompt")
        
        err = exc_info.value
        assert err.is_temporary is False
        assert err.status_code == 404
        assert "404 NOT FOUND" in str(err)
        assert mock_client.models.generate_content.call_count == 1


def test_gemini_429_rotates_immediately_to_next_key():
    model = GeminiModel(model="gemini-3.5-flash")
    keys = ["AIzaSyKeyA11111111111111111111111111111", "AIzaSyKeyB22222222222222222222222222222"]
    
    with patch.object(GeminiKeyManager, "get_keys", return_value=keys):
        GeminiKeyManager.rotator.set_keys(keys)
        
        call_keys = []
        def mock_generate_content(*args, **kwargs):
            active_key = model.api_key
            call_keys.append(active_key)
            if active_key == keys[0]:
                raise RuntimeError("429 RESOURCE_EXHAUSTED. Quota exceeded. Please retry after 10s.")
            resp = MagicMock()
            resp.text = '{"success": true}'
            resp.candidates = [MagicMock(finish_reason="STOP", content=MagicMock(parts=[MagicMock(text='{"success": true}')]))]
            resp.usage_metadata = MagicMock(prompt_token_count=10, candidates_token_count=5)
            return resp

        with patch("google.genai.Client") as mock_genai_client:
            instance = MagicMock()
            instance.models.generate_content.side_effect = mock_generate_content
            mock_genai_client.return_value = instance

            response = model.generate("test prompt")
            assert response.content == '{"success": true}'
            assert call_keys[0] == keys[0]
            assert call_keys[1] == keys[1]


def test_premise_architect_does_not_swallow_model_unavailable():
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = ModelUnavailableError("Model overloaded (503)", is_temporary=True, status_code=503)

    with pytest.raises(ModelUnavailableError):
        PremiseArchitect.generate(idea_text="A thrilling tale of space exploration", llm=mock_llm)

    assert mock_llm.generate.call_count == 1


def test_premise_architect_does_not_swallow_rate_limit():
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = RateLimitExhaustedError("Daily limit reached", wait_seconds=3600, is_daily=True)

    with pytest.raises(RateLimitExhaustedError):
        PremiseArchitect.generate(idea_text="A thrilling tale of space exploration", llm=mock_llm)

    assert mock_llm.generate.call_count == 1


def test_api_error_handler_status_codes():
    with app.test_request_context():
        # 503 Temporary unavailable
        err_503 = ModelUnavailableError("Google high demand spike", is_temporary=True, status_code=503)
        resp, code = _handle_llm_exception(err_503, "test")
        assert code == 503
        data = resp.get_json()
        assert data["error_type"] == "model_unavailable"
        assert data["is_temporary"] is True

        # 404 Model not found
        err_404 = ModelUnavailableError("Model retired", is_temporary=False, status_code=404)
        resp, code = _handle_llm_exception(err_404, "test")
        assert code == 404
        data = resp.get_json()
        assert data["error_type"] == "model_not_found"
        assert data["is_temporary"] is False

        # 429 Rate limit
        err_429 = RateLimitExhaustedError("Rate limit exceeded across all keys", wait_seconds=60.0, is_daily=False)
        resp, code = _handle_llm_exception(err_429, "test")
        assert code == 429
        data = resp.get_json()
        assert data["error_type"] == "rate_limit"
        assert data["wait_seconds"] == 60.0


def test_groq_quota_deferred_swaps_immediately_without_waiting():
    test_keys = ["gsk_key11111111111111111111111111111111111111111111111111", "gsk_key22222222222222222222222222222222222222222222222222"]
    with patch.object(GroqKeyManager, "get_keys", return_value=test_keys):
        GroqKeyManager.rotator.set_keys(test_keys)

        mock_scheduler = MagicMock()
        calls = []
        def mock_acquire(group, model, input_tokens, max_tokens, cancel, sleep, progress, max_wait):
            calls.append(group)
            if len(calls) == 1:
                # First key has 13901s wait (like account 3 in production!)
                from models.base import QuotaDeferred
                raise QuotaDeferred("provider tokens", 13901.9)
            # Second key acquires reservation successfully!
            reservation = MagicMock()
            reservation.id = "res-2"
            reservation.group = group
            reservation.model = model
            return reservation

        mock_scheduler.acquire.side_effect = mock_acquire

        model = GroqModel(model="qwen/qwen3.8-27b", scheduler=mock_scheduler)
        with patch.object(model, "_client_for") as mock_client_factory:
            client_inst = MagicMock()
            mock_client_factory.return_value = client_inst
            raw_mock = MagicMock()
            raw_mock.headers = {}
            resp_mock = MagicMock()
            resp_mock.choices = [MagicMock(finish_reason="stop", message=MagicMock(content="Hello world"))]
            raw_mock.parse.return_value = resp_mock
            client_inst.chat.completions.with_raw_response.create.return_value = raw_mock

            res = model.generate("test prompt")
            assert res.content == "Hello world"
            # Verify it failed over from key 1 to key 2 immediately!
            assert len(calls) == 2

