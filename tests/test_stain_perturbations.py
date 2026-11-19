import unittest
import numpy as np
from PIL import Image
from src.stain_perturbations import image_rng, perturb_stain


class StainTests(unittest.TestCase):
    def setUp(self):
        generator = np.random.default_rng(17)
        concentrations = generator.uniform(0.1, 2, (32, 32, 2))
        basis = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])
        self.image = Image.fromarray(np.uint8(255 * np.exp(-(concentrations @ basis))))

    def test_original_and_zero_strength_preserve_pixels(self):
        for method, strength in (("original", 0.1), ("hed", 0), ("macenko", 0)):
            actual = perturb_stain(self.image, method, strength, image_rng("a", 42))
            np.testing.assert_array_equal(actual, self.image)

    def test_methods_reproducible_and_change_stain(self):
        for method in ("hed", "macenko"):
            a = np.asarray(perturb_stain(self.image, method, 0.1, image_rng("a", 42)))
            b = np.asarray(perturb_stain(self.image, method, 0.1, image_rng("a", 42)))
            np.testing.assert_array_equal(a, b)
            self.assertEqual(a.shape, (32, 32, 3))
            self.assertGreater(np.abs(a.astype(float) - np.asarray(self.image)).mean(), 0.1)

    def test_macenko_cannot_silently_skip_blank_patch(self):
        with self.assertRaisesRegex(ValueError, "Insufficient tissue"):
            perturb_stain(Image.new("RGB", (32, 32), "white"), "macenko", 0.1, image_rng("a", 42))

    def test_same_slide_different_files_have_distinct_rng(self):
        self.assertNotEqual(image_rng("test/ADI/1_2_ADI.png", 42).random(),
                            image_rng("test/ADI/1_3_ADI.png", 42).random())


if __name__ == "__main__":
    unittest.main()
