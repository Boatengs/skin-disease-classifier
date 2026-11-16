"""Skin lesion classification on HAM10000 with lesion-level evaluation."""

CLASSES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
CLASS_NAMES = {
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "bcc": "Basal Cell Carcinoma",
    "akiec": "Actinic Keratosis",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "vasc": "Vascular Lesion",
}
MALIGNANT = ["mel", "bcc", "akiec"]
MEL = CLASSES.index("mel")

__all__ = ["CLASSES", "CLASS_NAMES", "MALIGNANT", "MEL"]
