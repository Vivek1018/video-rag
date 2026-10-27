from video_rag.language import language_from_filename, resolve_language


def test_filename_german_tag():
    assert language_from_filename("Clouds-Deutsch.mp3") == "de"


def test_filename_falls_back_to_default():
    lang, source = resolve_language("Gather.mp3", "Gather.mp3", default="en")
    assert lang == "en"
    assert source in {"filename", "metadata", "default"}
