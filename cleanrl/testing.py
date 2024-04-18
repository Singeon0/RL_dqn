import unittest
import numpy as np
from dqn import transform_array, explore_shape, extract_shape_positions, MedicalImageSegmentationEnv


class TestTransformArray(unittest.TestCase):

    def test_transform_array_with_even_length_input(self):
        input_array = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        array_mu_x, array_mu_y = transform_array(input_array)
        self.assertEqual(array_mu_x.shape, (2, 2, 1))
        self.assertEqual(array_mu_y.shape, (2, 2, 1))
        np.testing.assert_array_equal(array_mu_x, np.array([[[1], [3]], [[5], [7]]]))

    def test_transform_array_with_odd_length_input(self):
        input_array = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            transform_array(input_array)

    def test_transform_array_with_zero_length_input(self):
        input_array = np.array([])
        with self.assertRaises(ValueError):
            transform_array(input_array)

    def test_transform_array_with_large_input(self):
        input_array = [i for i in range(18)]
        array_mu_x, array_mu_y = transform_array(input_array)
        self.assertEqual(array_mu_x.shape, (3, 3, 1))
        self.assertEqual(array_mu_y.shape, (3, 3, 1))


class TestShapeExtraction(unittest.TestCase):

    def test_explore_shape_with_single_pixel_shape(self):
        mask = np.zeros((5, 5), dtype=np.int64)
        mask[2, 2] = 1
        start_pix = (2, 2)
        shape_pixels, temp = explore_shape(mask, start_pix)
        self.assertEqual(len(shape_pixels), 1)
        self.assertEqual(shape_pixels[0], start_pix)
        self.assertTrue(np.all(temp == 0))

    def test_explore_shape_with_multiple_pixel_shape(self):
        mask = np.zeros((5, 5), dtype=np.int64)
        mask[2, 2] = 1
        mask[2, 3] = 1
        start_pix = (2, 2)
        shape_pixels, temp = explore_shape(mask, start_pix)
        self.assertEqual(len(shape_pixels), 2)
        self.assertTrue((2, 2) in shape_pixels)
        self.assertTrue((2, 3) in shape_pixels)
        self.assertTrue(np.all(temp == 0))

    def test_extract_shape_positions_with_no_shapes(self):
        shape = np.zeros((5, 5), dtype=np.int64)
        shape_positions = extract_shape_positions(shape)
        self.assertEqual(len(shape_positions), 0)

    def test_extract_shape_positions_with_single_shape(self):
        shape = np.zeros((5, 5), dtype=np.int64)
        shape[2, 2] = 1
        shape_positions = extract_shape_positions(shape)
        self.assertEqual(len(shape_positions), 1)
        self.assertEqual(len(shape_positions[0]), 1)
        self.assertEqual(shape_positions[0][0], (2, 2))

    def test_extract_shape_positions_with_multiple_shapes(self):
        shape = np.zeros((5, 5), dtype=np.int64)
        shape[2, 2] = 1
        shape[4, 4] = 1
        shape_positions = extract_shape_positions(shape)
        self.assertEqual(len(shape_positions), 2)
        self.assertEqual(len(shape_positions[0]), 1)
        self.assertEqual(len(shape_positions[1]), 1)
        self.assertTrue((2, 2) in shape_positions[0] or (2, 2) in shape_positions[1])
        self.assertTrue((4, 4) in shape_positions[0] or (4, 4) in shape_positions[1])


class IoUComputationTests(unittest.TestCase):

    def setUp(self):
        self.env = MedicalImageSegmentationEnv('utah_test_set.h5', 4, 10, 0.5)

    def test_identical_masks_produce_iou_of_one(self):
        mask1 = np.array([[0, 1], [1, 0]])
        mask2 = np.array([[0, 1], [1, 0]])
        iou = self.env._compute_iou(mask1, mask2)
        self.assertEqual(iou, 1.0)

    def test_disjoint_masks_produce_iou_of_zero(self):
        mask1 = np.array([[0, 1], [1, 0]])
        mask2 = np.array([[1, 0], [0, 1]])
        iou = self.env._compute_iou(mask1, mask2)
        self.assertEqual(iou, 0.0)

    def test_overlapping_masks_produce_iou_between_zero_and_one(self):
        mask1 = np.array([[0, 1], [1, 0]])
        mask2 = np.array([[0, 1], [1, 1]])
        iou = self.env._compute_iou(mask1, mask2)
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)

if __name__ == '__main__':
    unittest.main()
