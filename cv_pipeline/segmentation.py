from pathlib import Path

import cv2
import numpy as np

from sklearn.cluster import KMeans
from skimage.segmentation import slic
from ultralytics import SAM


# ---------------------------------------------------------
# MODEL PATH
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAM_MODEL_PATH = PROJECT_ROOT / "models" / "sam2_b.pt"


# ---------------------------------------------------------
# LOAD SAM2 MODEL
# ---------------------------------------------------------

sam_model = SAM(str(SAM_MODEL_PATH))


# ---------------------------------------------------------
# HELPER: EXTRACT FIRST SAM MASK
# ---------------------------------------------------------

def _extract_first_mask(results, image_shape):
    """
    Extract the first mask returned by SAM/SAM2
    and resize it to the original image dimensions.
    """

    if not results:
        return None

    result = results[0]

    if result.masks is None:
        return None

    masks = result.masks.data.cpu().numpy()

    if len(masks) == 0:
        return None

    mask = masks[0].astype(np.uint8)

    height, width = image_shape[:2]

    mask = cv2.resize(
        mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )

    mask = mask > 0

    return mask


# ---------------------------------------------------------
# STAGE 1: SAM2 LEAF SEGMENTATION
# ---------------------------------------------------------

def segment_leaf(image, bbox):
    """
    Use the YOLO bounding box as a SAM2 bounding-box prompt.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image.

    bbox : list
        YOLO bounding box [x1, y1, x2, y2].

    Returns
    -------
    np.ndarray or None
        Boolean leaf mask.
    """

    results = sam_model(
        image,
        bboxes=[bbox],
        verbose=False
    )

    leaf_mask = _extract_first_mask(
        results,
        image.shape
    )

    return leaf_mask


# ---------------------------------------------------------
# STAGE 2: SLIC SUPERPIXELS
# ---------------------------------------------------------

def generate_superpixels(
    image_rgb,
    n_segments=200,
    compactness=10
):
    """
    Generate SLIC superpixels.

    Frozen notebook parameters:
        n_segments = 200
        compactness = 10
        start_label = 1
    """

    segments = slic(
        image_rgb,
        n_segments=n_segments,
        compactness=compactness,
        start_label=1
    )

    return segments


# ---------------------------------------------------------
# STAGE 3: LAB FEATURES FOR SUPERPIXELS
# ---------------------------------------------------------

def compute_superpixel_lab_features(
    image_rgb,
    segments,
    leaf_mask
):
    """
    Calculate mean Lab values for superpixels
    that overlap the detected leaf.
    """

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    image_lab = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2LAB
    )

    labels = np.unique(segments)

    features = []
    valid_labels = []

    for label in labels:

        region = segments == label

        # Only keep superpixels that overlap leaf
        if not np.any(region & leaf_mask):
            continue

        pixels = image_lab[
            region & leaf_mask
        ]

        if len(pixels) == 0:
            continue

        mean_lab = pixels.mean(axis=0)

        features.append(mean_lab)
        valid_labels.append(label)

    if len(features) == 0:
        return (
            np.empty((0, 3)),
            np.array([], dtype=int)
        )

    return (
        np.array(features),
        np.array(valid_labels)
    )


# ---------------------------------------------------------
# STAGE 4: K-MEANS CLUSTERING
# ---------------------------------------------------------

def cluster_superpixels(
    features,
    valid_labels,
    segments,
    leaf_mask,
    n_clusters=4
):
    """
    Cluster superpixels using K-Means.

    Frozen notebook parameters:
        n_clusters = 4
        random_state = 42
        n_init = 10
    """

    if len(features) == 0:
        return None, None

    # Avoid requesting more clusters than samples
    actual_clusters = min(
        n_clusters,
        len(features)
    )

    kmeans = KMeans(
        n_clusters=actual_clusters,
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(features)

    cluster_map = np.full(
        segments.shape,
        -1,
        dtype=np.int32
    )

    for label, cluster in zip(
        valid_labels,
        cluster_labels
    ):
        cluster_map[
            segments == label
        ] = cluster

    # Keep only the detected leaf
    cluster_map[
        ~leaf_mask
    ] = -1

    return cluster_map, kmeans


# ---------------------------------------------------------
# STAGE 5: CANDIDATE DISEASE MASK
# ---------------------------------------------------------

def create_candidate_mask(
    cluster_map,
    leaf_mask,
    candidate_cluster=1
):
    """
    Select the candidate disease cluster.

    IMPORTANT:
    candidate_cluster = 1 is intentionally preserved
    from the frozen research notebook.
    """

    candidate_mask = (
        (cluster_map == candidate_cluster)
        & leaf_mask
    )

    return candidate_mask


# ---------------------------------------------------------
# STAGE 6: POSITIVE POINT
# ---------------------------------------------------------

def find_positive_point(candidate_mask):
    """
    Find the deepest/interior point of the candidate
    region using a distance transform.
    """

    candidate_uint8 = (
        candidate_mask.astype(np.uint8) * 255
    )

    if not np.any(candidate_mask):
        return None

    distance = cv2.distanceTransform(
        candidate_uint8,
        cv2.DIST_L2,
        5
    )

    _, max_value, _, max_location = cv2.minMaxLoc(
        distance
    )

    if max_value <= 0:
        return None

    x, y = max_location

    return [int(x), int(y)]


# ---------------------------------------------------------
# STAGE 7: HEALTHY NEGATIVE POINT
# ---------------------------------------------------------

def find_negative_point(
    leaf_mask,
    candidate_mask
):
    """
    Find a point inside the leaf but outside
    the candidate disease region.
    """

    healthy_region = (
        leaf_mask
        & (~candidate_mask)
    )

    healthy_uint8 = (
        healthy_region.astype(np.uint8) * 255
    )

    if not np.any(healthy_region):
        return None

    distance = cv2.distanceTransform(
        healthy_uint8,
        cv2.DIST_L2,
        5
    )

    _, max_value, _, max_location = cv2.minMaxLoc(
        distance
    )

    if max_value <= 0:
        return None

    x, y = max_location

    return [int(x), int(y)]


# ---------------------------------------------------------
# STAGE 8: SAM2 POINT PROMPT
# ---------------------------------------------------------

def segment_disease_with_points(
    image,
    positive_point,
    negative_point,
    leaf_mask
):
    """
    Use the candidate-region positive point and healthy
    negative point as SAM2 prompts.

    The notebook selects the smallest valid SAM2 mask
    containing the positive point.
    """

    if positive_point is None:
        return None

    if negative_point is None:
        return None

    points = [
        positive_point,
        negative_point
    ]

    labels = [
        1,
        0
    ]

    results = sam_model(
        image,
        points=points,
        labels=labels,
        verbose=False
    )

    if not results:
        return None

    result = results[0]

    if result.masks is None:
        return None

    sam_masks_data = (
        result.masks.data
        .cpu()
        .numpy()
    )

    if len(sam_masks_data) == 0:
        return None

    # -----------------------------------------------------
    # Find masks containing the positive point
    # -----------------------------------------------------

    px, py = positive_point

    valid_masks = []

    for i, mask in enumerate(
        sam_masks_data
    ):

        height, width = mask.shape

        if (
            0 <= py < height
            and 0 <= px < width
            and mask[py, px] > 0
        ):
            valid_masks.append(i)

    if not valid_masks:
        return None

    # -----------------------------------------------------
    # Frozen notebook selection logic:
    # choose smallest valid mask
    # -----------------------------------------------------

    best_mask_idx = min(
        valid_masks,
        key=lambda i: np.sum(
            sam_masks_data[i] > 0
        )
    )

    disease_mask = (
        sam_masks_data[best_mask_idx] > 0
    )

    # Resize to original image
    height, width = image.shape[:2]

    disease_mask = cv2.resize(
        disease_mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )

    disease_mask = (
        disease_mask.astype(bool)
        & leaf_mask
    )

    return disease_mask