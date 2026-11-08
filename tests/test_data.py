from PIL import Image

from src.food11.data import CLASSES, category_from_filename, resize_image


def test_category_from_filename():
    assert category_from_filename("0_12.jpg") == "Egg"
    assert category_from_filename("10_3.jpg") == "Vegetable-Fruit"


def test_resize_image_shape_and_mode():
    with Image.new("L", (512, 384)) as image:
        resized = resize_image(image)
    assert resized.size == (128, 128)
    assert resized.mode == "RGB"


def test_eleven_classes():
    assert len(CLASSES) == len(set(CLASSES)) == 11
