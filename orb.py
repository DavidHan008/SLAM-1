def orb_detector_using_tiles(image, max_number_of_kp = 20, overlap_div = 2, height_div = 5, width_div = 10):
    def get_kps(x, y, h, w):
        impatch = image[y:y + h, x:x + w]
        keypoints, descriptors = orb_extraction_detect(impatch)
        for pt in keypoints:
            pt.pt = (pt.pt[0] + x, pt.pt[1] + y)
        if len(keypoints) == 0:
            return [], []
        return keypoints, descriptors
    h, w = image.shape
    tile_h = int(h/height_div)
    tile_w = int(w/width_div)
    kp_list = []
    des = []
    for y in range(0, h-tile_h, tile_h):
        for x in range(0, w-tile_w, tile_w):
            kp1, des1 = get_kps(x, y, int(tile_h+tile_h/overlap_div), int(tile_w+tile_w/overlap_div))
            kp_list.extend(kp1)
            des.extend(des1)
    des = [kp for sublist in des for kp in sublist]
    des = np.reshape(des, (-1, 32))
    return kp_list, des


def orb_extraction_detect(img):
    """FAST corners at 8 scale levels with a scale factor of 1.2.
    For image resolutions from 512×384 to 752×480 pixels we found suitable to extract 1000 corners,
    for higher resolutions, as the 1241 × 376 in the KITTI dataset [40] we extract 2000 corners"""

    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures = 40, scaleFactor=1.2)
    # find the keypoints with ORB
    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)

    return kp, des


def orb_extraction_compute(img, kp):
    orb = cv2.ORB_create(nfeatures = 50, scaleFactor=1.2)
    kp, des = orb.compute(img, kp)
    return kp, des