import cv2
import numpy as np

from .detection import detect_crop

from .segmentation import (
    segment_leaf,
    generate_superpixels,
    compute_superpixel_lab_features,
    cluster_superpixels,
    create_candidate_mask,
    find_positive_point,
    find_negative_point,
    segment_disease_with_points,
)


# ---------------------------------------------------------
# MAIN CROP DISEASE ANALYSIS
# ---------------------------------------------------------

def analyze_disease(
    image,
    candidate_cluster=1
):
    """
    Run the complete frozen CropGuard v0.1 pipeline.

    Pipeline:

        Image
          ↓
        YOLO detection
          ↓
        SAM2 leaf segmentation
          ↓
        SLIC superpixels
          ↓
        Lab features
          ↓
        K-Means
          ↓
        Candidate cluster
          ↓
        Positive + negative points
          ↓
        SAM2 point prompting
          ↓
        Disease mask
          ↓
        Affected leaf area %

    Parameters
    ----------
    image : np.ndarray
        Input BGR image.

    candidate_cluster : int
        Frozen candidate K-Means cluster.
        Default = 1.

    Returns
    -------
    dict
        Complete analysis result.
    """

    # -----------------------------------------------------
    # 1. YOLO DETECTION
    # -----------------------------------------------------

    detection = detect_crop(image)

    if detection is None:
        return {
            "success": False,
            "error": "No crop/disease detection was found."
        }

    bbox = detection["bbox"]

    # -----------------------------------------------------
    # 2. CONVERT IMAGE FOR SLIC/LAB
    # -----------------------------------------------------

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # -----------------------------------------------------
    # 3. SAM2 LEAF SEGMENTATION
    # -----------------------------------------------------

    leaf_mask = segment_leaf(
        image,
        bbox
    )

    if leaf_mask is None:
        return {
            "success": False,
            "error": "SAM2 could not segment the detected leaf.",
            "detection": detection
        }

    # -----------------------------------------------------
    # 4. SLIC SUPERPIXELS
    # -----------------------------------------------------

    segments = generate_superpixels(
        image_rgb,
        n_segments=200,
        compactness=10
    )

    # -----------------------------------------------------
    # 5. LAB FEATURES
    # -----------------------------------------------------

    features, valid_labels = (
        compute_superpixel_lab_features(
            image_rgb,
            segments,
            leaf_mask
        )
    )

    if len(features) == 0:
        return {
            "success": False,
            "error": "No valid leaf superpixels were found.",
            "detection": detection,
            "leaf_mask": leaf_mask
        }

    # -----------------------------------------------------
    # 6. K-MEANS
    # -----------------------------------------------------

    cluster_map, kmeans = (
        cluster_superpixels(
            features,
            valid_labels,
            segments,
            leaf_mask,
            n_clusters=4
        )
    )

    if cluster_map is None:
        return {
            "success": False,
            "error": "K-Means clustering could not be completed.",
            "detection": detection,
            "leaf_mask": leaf_mask
        }

    # -----------------------------------------------------
    # 7. CANDIDATE DISEASE REGION
    # -----------------------------------------------------

    candidate_mask = create_candidate_mask(
        cluster_map,
        leaf_mask,
        candidate_cluster=candidate_cluster
    )

    # -----------------------------------------------------
    # 8. POSITIVE POINT
    # -----------------------------------------------------

    positive_point = find_positive_point(
        candidate_mask
    )

    # -----------------------------------------------------
    # 9. HEALTHY NEGATIVE POINT
    # -----------------------------------------------------

    negative_point = find_negative_point(
        leaf_mask,
        candidate_mask
    )

    # -----------------------------------------------------
    # 10. SAM2 DISEASE SEGMENTATION
    # -----------------------------------------------------

    disease_mask = segment_disease_with_points(
        image,
        positive_point,
        negative_point,
        leaf_mask
    )

    # -----------------------------------------------------
    # 11. AFFECTED AREA
    # -----------------------------------------------------

    leaf_pixels = int(
        np.sum(leaf_mask)
    )

    if disease_mask is None:

        disease_pixels = 0
        affected_area_percent = 0.0

    else:

        disease_mask = (
            disease_mask
            & leaf_mask
        )

        disease_pixels = int(
            np.sum(disease_mask)
        )

        if leaf_pixels > 0:

            affected_area_percent = (
                disease_pixels
                / leaf_pixels
            ) * 100.0

        else:

            affected_area_percent = 0.0

    # -----------------------------------------------------
    # 12. RETURN COMPLETE RESULT
    # -----------------------------------------------------

    return {
        "success": True,

        # Original image
        "image": image,

        # YOLO
        "detection": detection,

        # Segmentation
        "leaf_mask": leaf_mask,
        "candidate_mask": candidate_mask,
        "disease_mask": disease_mask,

        # Prompt points
        "positive_point": positive_point,
        "negative_point": negative_point,

        # Statistics
        "leaf_pixels": leaf_pixels,
        "disease_pixels": disease_pixels,
        "affected_area_percent": float(
            affected_area_percent
        ),

        # Research information
        "candidate_cluster": candidate_cluster,
        "cluster_map": cluster_map,
    }