import re


def mask_account_number(text: str) -> str:
    """
    Mask account numbers before sending sensitive
    information to external AI services.
    """

    pattern = r"\b\d{8,18}\b"

    return re.sub(
        pattern,
        "[ACCOUNT_NUMBER]",
        text
    )


def mask_phone_number(text: str) -> str:
    """
    Mask common phone-number patterns.
    """

    pattern = r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b"

    return re.sub(
        pattern,
        "[PHONE_NUMBER]",
        text
    )


def mask_pii(text: str) -> str:
    """
    Apply basic PII masking.
    """

    text = mask_account_number(text)
    text = mask_phone_number(text)

    return text