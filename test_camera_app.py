import unittest

import cv2
import numpy as np

from camera_app import (
    camera_region,
    format_digit_string,
    preprocess_digit,
    segment_digits,
)


class PreprocessDigitTests(unittest.TestCase):
    def test_preprocesses_dark_digit_into_mnist_input(self):
        image = np.full((160, 120), 255, dtype=np.uint8)
        cv2.line(image, (35, 30), (80, 30), 0, 8)
        cv2.line(image, (80, 30), (45, 80), 0, 8)
        cv2.line(image, (45, 80), (80, 130), 0, 8)

        result = preprocess_digit(image)

        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (784, 1))
        self.assertEqual(result.dtype, np.float32)
        self.assertGreater(float(result.max()), 0)
        self.assertLessEqual(float(result.max()), 1)

    def test_rejects_blank_image(self):
        image = np.full((100, 100), 255, dtype=np.uint8)

        self.assertIsNone(preprocess_digit(image))

    def test_camera_region_selects_center_of_frame(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[15:85, 20:80] = (10, 100, 200)
        cv2.line(frame, (40, 30), (60, 30), (0, 0, 0), 5)
        cv2.line(frame, (60, 30), (45, 50), (0, 0, 0), 5)
        cv2.line(frame, (45, 50), (60, 70), (0, 0, 0), 5)

        region = camera_region(frame)

        self.assertEqual(region.shape, (70, 60))
        self.assertEqual(region.dtype, np.uint8)
        self.assertGreater(int(region.mean()), 0)
        self.assertIsNotNone(preprocess_digit(region))

    def test_segments_separated_digits_from_left_to_right(self):
        image = np.full((120, 280), 255, dtype=np.uint8)
        cv2.putText(image, "3242", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 5)

        crops = segment_digits(image)

        self.assertEqual(len(crops), 4)
        self.assertTrue(all(preprocess_digit(crop) is not None for crop in crops))

    def test_ignores_small_comma_and_groups_result(self):
        image = np.full((140, 320), 255, dtype=np.uint8)
        cv2.putText(image, "3,242", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 5)

        crops = segment_digits(image)

        self.assertEqual(len(crops), 4)
        self.assertEqual(format_digit_string("3242"), "3,242")
        self.assertEqual(format_digit_string("0012345"), "0,012,345")


if __name__ == "__main__":
    unittest.main()
