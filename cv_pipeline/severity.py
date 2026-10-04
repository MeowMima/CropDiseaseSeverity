# ---------------------------------------------------------
# PROTOTYPE SEVERITY CLASSIFICATION
# ---------------------------------------------------------

def classify_severity(affected_area):
    """
    Prototype severity classification based on
    affected leaf area percentage.

    Thresholds are preserved exactly from
    the research notebook.

    IMPORTANT:
    These thresholds are NOT scientifically validated
    agricultural severity standards.
    """

    if affected_area < 10:
        return "Low"

    elif affected_area < 25:
        return "Moderate"

    elif affected_area < 50:
        return "High"

    else:
        return "Severe"