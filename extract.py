import numpy as np
import cv2
import os

def extract(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    # reshape points
    points = points.reshape(4, 2).tolist()
    points = [[point[0] * image.shape[1], point[1] * image.shape[0]] for point in points]
    points = np.array(points, dtype=np.float32)

    # compute height and with of plate based on height
    height = np.int64(abs(points[0, 1] - points[0, 0]))
    width = np.int64(4.5*height)

    # desired size of plate (point are 4 corners)
    points2 = np.array([[0, 0],
                        [width, 0],
                        [width, height],
                        [0, height]]).astype(np.float32)

    # homography to convert plate to corresponding desired size
    homography = cv2.getPerspectiveTransform(points, points2)
    output_size = (width, height)
    warp = cv2.warpPerspective(image, homography, output_size)

    return warp


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

        result = extract(in_image, np.array(coordinates, dtype=np.float32))
        cv2.imshow('extracted plate image', result)
        cv2.waitKey()
