import numpy as np
import cv2
import os

def mask(image: np.ndarray, points: np.ndarray, cover: np.ndarray) -> np.ndarray:
    # reshape points
    points = points.reshape(4, 2).tolist()
    points = [[point[0] * image.shape[1], point[1] * image.shape[0]] for point in points]
    points = np.array(points, dtype=np.float32)

    # find homography of cover shape to points
    cover_points = np.array([[0, 0],
                             [cover.shape[1], 0],
                             [cover.shape[1], cover.shape[0]],
                             [0, cover.shape[0]]]).astype(np.float32)

    homography = cv2.getPerspectiveTransform(cover_points, points)
    output_size = (image.shape[1], image.shape[0])
    warp = cv2.warpPerspective(cover, homography, output_size)

    img_copy = image.copy()
    img_copy[warp[:, :, 0] != 0] = [0, 0, 0]

    # stack two images to have (Height, Width, 3, 2)
    t = np.stack((warp, img_copy), axis=-1)
    out = np.max(t, axis=-1)

    return out


if __name__ == '__main__':
    test_images = ['test_plates//1a4ea979-day_13562', 'test_plates//1f193783-day_00271',
                   'test_plates//1ef77e57-day_07335']
    for i in range(len(test_images)):
        img = os.path.join(os.getcwd(), test_images[i])

        with open(img + '.txt') as file:
            for line in file:
                coordinates = line.rstrip().split(' ')

        coordinates.pop(0)
        in_image = cv2.imread(img + '.jpg')
        cover_image = cv2.imread('kntu.jpg')

        result = mask(in_image, np.array(coordinates, dtype=np.float32), cover_image)
        cv2.imshow('masked plate image', result)
        cv2.waitKey()
