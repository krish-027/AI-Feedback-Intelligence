def clean_text(text: str) -> str:
    """
    Basic text cleaning.

    Detailed preprocessing will be implemented later.
    """

    if not text:
        return ""

    text = text.replace("\x00", " ")

    return " ".join(text.split())
