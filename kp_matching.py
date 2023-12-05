import rasterio
from rasterio.plot import reshape_as_image
import rasterio.mask

import numpy as np
import cv2 as cv
# import matplotlib.pyplot as plt


def kp_match(data_path1: str, data_path2: str, output_path: str) -> None:
    """
    This function performs keypoint matching on two satellite images
    It calculates the Vegetation Index (B8-B4)/(B8+B4) for each image
    and uses keypoint detection and matching algorithms to find similarities.

    Parameters:
    data_path1 (str): The file path to the first satellite image. Path to _TCI.jp2 file.
    data_path2 (str): The file path to the second satellite image. Path to _TCI.jp2 file.
    But keep the standard directory structure of Sentinel-2 as we will use other data from it.

    output_path(str): The path to result file. Image in JPG format.

    Returns:
    None: The function does not return anything.
     Instead, it saves and displays an image showing the matched keypoints between the two input images.
    """
    # Replacing the file suffix to get paths for Band 4 (red) and Band 8 (NIR)
    data_path1_b4 = data_path1.replace("TCI", "B04")
    data_path1_b8 = data_path1.replace("TCI", "B08")

    # Opening and reading the true color image (TCI) from the first data path
    with rasterio.open(data_path1, "r", driver="JP2OpenJPEG") as src:
        raster_img1 = src.read()
    raster_img1 = reshape_as_image(raster_img1)

    # Opening and reading Band 4 and Band 8 images for the first data path
    with rasterio.open(data_path1_b4, "r", driver="JP2OpenJPEG") as src:
        raster_img1_b4 = src.read()
    raster_img1_b4 = reshape_as_image(raster_img1_b4)
    with rasterio.open(data_path1_b8, "r", driver="JP2OpenJPEG") as src:
        raster_img1_b8 = src.read()
    raster_img1_b8 = reshape_as_image(raster_img1_b8)

    # Calculating the Vegetation Index (B8-B4)/(B8+B4) for the first image
    raster_img1_result = (raster_img1_b8 - raster_img1_b4) / (raster_img1_b8 + raster_img1_b4)

    # Repeating the process for the second data path
    data_path2_b4 = data_path2.replace("TCI", "B04")
    data_path2_b8 = data_path2.replace("TCI", "B08")
    with rasterio.open(data_path2, "r", driver="JP2OpenJPEG") as src:
        raster_img2 = src.read()
    raster_img2 = reshape_as_image(raster_img2)

    with rasterio.open(data_path2_b4, "r", driver="JP2OpenJPEG") as src:
        raster_img2_b4 = src.read()
    raster_img2_b4 = reshape_as_image(raster_img2_b4)

    with rasterio.open(data_path2_b8, "r", driver="JP2OpenJPEG") as src:
        raster_img2_b8 = src.read()
    raster_img2_b8 = reshape_as_image(raster_img2_b8)

    # Calculating the Vegetation Index (B8-B4)/(B8+B4) for the second image
    raster_img2_result = (raster_img2_b8 - raster_img2_b4) / (raster_img2_b8 + raster_img2_b4)

    # Normalizing the results to prepare for keypoint detection
    raster_img1_result_norm = np.uint8(cv.normalize(raster_img1_result, None, 0, 255, cv.NORM_MINMAX))
    raster_img2_result_norm = np.uint8(cv.normalize(raster_img2_result, None, 0, 255, cv.NORM_MINMAX))

    # Initiate keypoint detectors
    akaze = cv.AKAZE_create()
    sift = cv.SIFT_create()

    # Detect keypoints and compute descriptors using AKAZE + SIFT
    kp1 = akaze.detect(raster_img1_result_norm, None)
    kp2 = akaze.detect(raster_img2_result_norm, None)
    kp1, des1 = sift.compute(raster_img1_result_norm, kp1)
    kp2, des2 = sift.compute(raster_img2_result_norm, kp2)

    # Using BFMatcher to match keypoints
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test to find good matches
    good = []
    threshold = 0.5  # The coef was adjusted during the solution process to reduce the number of incorrect connections.
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good.append([m])

    # Drawing the matches
    img3 = cv.drawMatchesKnn(
        raster_img1, kp1,
        raster_img2, kp2,
        good,
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Saving and displaying the result
    cv.imwrite(output_path, img3)
    # plt.imshow(img3), plt.show()


if __name__ == "__main__":
    path_summer = ("archive/S2A_MSIL1C_20160830T083602_N0204_R064_T36UYA_20160830T083600"
                   "/S2A_MSIL1C_20160830T083602_N0204_R064_T36UYA_20160830T083600.SAFE/GRANULE"
                   "/L1C_T36UYA_A006210_20160830T083600/IMG_DATA/T36UYA_20160830T083602_TCI.jp2")
    path_autumn = ("archive/S2A_MSIL1C_20190904T083601_N0208_R064_T36UYA_20190904T110155"
                   "/S2A_MSIL1C_20190904T083601_N0208_R064_T36UYA_20190904T110155.SAFE/GRANULE"
                   "/L1C_T36UYA_A021940_20190904T084432/IMG_DATA/T36UYA_20190904T083601_TCI.jp2")
    # path_cloud = ("archive/S2A_MSIL1C_20190318T083701_N0207_R064_T36UYA_20190318T122410"
    #               "/S2A_MSIL1C_20190318T083701_N0207_R064_T36UYA_20190318T122410.SAFE/GRANULE"
    #               "/L1C_T36UYA_A019509_20190318T083954/IMG_DATA/T36UYA_20190318T083701_TCI.jp2")
    # path_snow = ("archive/S2A_MSIL1C_20161121T085252_N0204_R107_T36UYA_20161121T085252"
    #              "/S2A_MSIL1C_20161121T085252_N0204_R107_T36UYA_20161121T085252.SAFE/GRANULE"
    #              "/L1C_T36UYA_A007397_20161121T085252/IMG_DATA/T36UYA_20161121T085252_TCI.jp2")
    # path_other_image = ("archive/S2A_MSIL1C_20180810T083601_N0206_R064_T36UXA_20180810T124435"
    #                     "/S2A_MSIL1C_20180810T083601_N0206_R064_T36UXA_20180810T124435.SAFE/GRANULE"
    #                     "/L1C_T36UXA_A016363_20180810T084438/IMG_DATA/T36UXA_20180810T083601_TCI.jp2")
    kp_match(path_summer, path_autumn, "result.jpg")
