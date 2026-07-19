"""Unit tests for execute_ai_trades idempotency helpers."""

from execute_ai_trades import should_skip_idempotent


def test_skip_when_same_recommendation_id():
    prior = {
        "recommendation_id": "abc-123",
        "action_taken": "HOLD",
        "decision": "SELL",
    }
    assert should_skip_idempotent(prior, current_reco_id="abc-123", current_decision="BUY") is True


def test_rerun_when_decision_changed_from_sell_to_buy():
    prior = {
        "recommendation_id": "old-id",
        "action_taken": "HOLD",
        "decision": "SELL",
    }
    assert should_skip_idempotent(prior, current_reco_id="new-id", current_decision="BUY") is False


def test_skip_when_already_bought_same_decision():
    prior = {
        "recommendation_id": "old-id",
        "action_taken": "BUY",
        "decision": "BUY",
    }
    assert should_skip_idempotent(prior, current_reco_id="new-id", current_decision="BUY") is True


def test_rerun_when_prior_skip_same_decision():
    prior = {
        "recommendation_id": "old-id",
        "action_taken": "SKIP",
        "decision": "BUY",
    }
    assert should_skip_idempotent(prior, current_reco_id="new-id", current_decision="BUY") is False
