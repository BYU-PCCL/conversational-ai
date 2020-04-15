"""Tests for the ``conversational_ai.chat`` module."""

from conversational_ai.chat import _postprocess_response


def test_postprocess_response() -> None:
    """Tests ``_postprocess_response``."""
    for input_txt, expected_result in [
        (
            "AdLab I need to start working on a project im interested in so Im gonna "
            "end the chat! speaker2> What are you up to?",
            "AdLab I need to start working on a project im interested in so Im gonna "
            "end the chat!",
        ),
        ("I don't think so. Dang it.", "I don't think so. Dang it."),
        ("Hello :) speaker2> Hi!", "Hello :)"),
        (
            "speaker2> So what a cool-me-do would you say that? speaker1> I'm not that "
            "familiar with LesCola",
            "So what a cool-me-do would you say that?",
        ),
        (
            "hahaha am I just going to take that as a no.... Hope you have a good day "
            "speaker2> i am in my ",
            "hahaha am I just going to take that as a no.... Hope you have a good day",
        ),
    ]:
        actual_result = _postprocess_response(input_txt, ["speaker1>", "speaker2>"])
        assert actual_result == expected_result
