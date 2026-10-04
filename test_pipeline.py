import cv2

from cv_pipeline import (
    analyze_disease,
    classify_severity,
)


# ---------------------------------------------------------
# TEST IMAGE
# ---------------------------------------------------------

IMAGE_PATH = (
    "test_images/286_1600.jpg"
)


# ---------------------------------------------------------
# LOAD IMAGE
# ---------------------------------------------------------

image = cv2.imread(
    IMAGE_PATH
)

if image is None:
    raise FileNotFoundError(
        f"Could not read image: {IMAGE_PATH}"
    )


print("\n" + "=" * 60)
print("KRISHI SAHYOG - PIPELINE TEST")
print("=" * 60)


# ---------------------------------------------------------
# RUN PIPELINE
# ---------------------------------------------------------

result = analyze_disease(
    image
)


# ---------------------------------------------------------
# CHECK RESULT
# ---------------------------------------------------------

if not result["success"]:

    print("\nPipeline failed.")
    print(
        "Reason:",
        result["error"]
    )

    raise SystemExit


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

detection = result["detection"]

print("\nYOLO Detection")
print("-" * 40)

print(
    "Class:",
    detection["class_name"]
)

print(
    "Confidence:",
    round(
        detection["confidence"],
        4
    )
)

print(
    "Bounding Box:",
    detection["bbox"]
)


print("\nSegmentation")
print("-" * 40)

print(
    "Leaf Pixels:",
    result["leaf_pixels"]
)

print(
    "Disease Pixels:",
    result["disease_pixels"]
)


print("\nAffected Area")
print("-" * 40)

affected_area = (
    result["affected_area_percent"]
)

print(
    "Affected Leaf Area:",
    round(
        affected_area,
        2
    ),
    "%"
)


# ---------------------------------------------------------
# SEVERITY
# ---------------------------------------------------------

severity = classify_severity(
    affected_area
)

print(
    "Prototype Severity:",
    severity
)


print("\n" + "=" * 60)
print("PIPELINE TEST COMPLETE")
print("=" * 60)