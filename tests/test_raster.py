import io

from PIL import Image

from im2vec.raster import flatten_to_rgb


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_transparent_rgba_flattens_to_white_not_black():
    # cairosvg stores (0,0,0) under fully transparent pixels; a naive
    # .convert("RGB") would turn the whole background black.
    img = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    out = flatten_to_rgb(img)
    assert out.mode == "RGB"
    assert out.getpixel((0, 0)) == (255, 255, 255)


def test_opaque_pixels_survive_flattening():
    img = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    img.putpixel((3, 3), (255, 0, 0, 255))
    out = flatten_to_rgb(img)
    assert out.getpixel((3, 3)) == (255, 0, 0)
    assert out.getpixel((0, 0)) == (255, 255, 255)


def test_custom_background_colour():
    img = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    out = flatten_to_rgb(img, background=(0, 0, 255))
    assert out.getpixel((0, 0)) == (0, 0, 255)


def test_palette_image_with_transparency():
    rgba = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    rgba.putpixel((1, 1), (0, 255, 0, 255))
    pal = rgba.convert("P", palette=Image.ADAPTIVE)
    pal.info["transparency"] = 0
    out = flatten_to_rgb(pal)
    assert out.mode == "RGB"


def test_already_rgb_passes_through():
    img = Image.new("RGB", (4, 4), (12, 34, 56))
    out = flatten_to_rgb(img)
    assert out.mode == "RGB"
    assert out.getpixel((0, 0)) == (12, 34, 56)


def test_greyscale_converts_to_rgb():
    img = Image.new("L", (4, 4), 128)
    out = flatten_to_rgb(img)
    assert out.mode == "RGB"
    assert out.getpixel((0, 0)) == (128, 128, 128)
