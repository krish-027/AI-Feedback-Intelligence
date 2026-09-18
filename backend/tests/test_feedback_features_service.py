from backend.services.feedback_features_service import (
    FeedbackFeaturesService,
)


def sample_feedback() -> str:
    return """
    Customer Feedback Form
    Employee Service Feedback
    Feedback Criteria Rating
    Employee greeted you politely 5
    Employee listened carefully to your concerns 2
    Employee explained banking services clearly 4
    Employee behaved professionally 3
    Employee resolved your issue efficiently 5
    Overall satisfaction with employee behavior 5
    Additional Comments
    Overall good experience.
    Recommendation: Yes
    """


def test_extracts_structured_fields() -> None:
    service = FeedbackFeaturesService()

    features = service.extract(
        sample_feedback()
    )

    assert features.greeting_rating == 5
    assert features.listening_rating == 2
    assert features.explanation_rating == 4
    assert features.professionalism_rating == 3
    assert features.resolution_rating == 5
    assert features.overall_satisfaction == 5
    assert features.recommendation == "Yes"


def test_calculates_derived_features() -> None:
    service = FeedbackFeaturesService()

    features = service.extract(
        sample_feedback()
    )

    assert features.average_service_rating == 3.8
    assert features.minimum_service_rating == 2
    assert features.low_rating_count == 1
    assert features.high_rating_count == 3


def test_prompt_summary_contains_structured_values() -> None:
    service = FeedbackFeaturesService()

    features = service.extract(
        sample_feedback()
    )

    summary = features.to_prompt_text()

    assert "Overall satisfaction: 5/5" in summary
    assert "Recommendation: Yes" in summary
    assert "Average service rating: 3.80/5" in summary


def test_invalid_feedback_is_rejected() -> None:
    service = FeedbackFeaturesService()

    try:
        service.extract(
            "This is not a valid feedback form."
        )
        assert False
    except ValueError:
        assert True