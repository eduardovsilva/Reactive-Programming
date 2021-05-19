from rx import of, pipe, operators as ops, create
from PIL import Image
from skimage import io
import numpy as np
from matplotlib import pyplot as plt


def distance(a, b):
    from math import sqrt
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def gaussian_filter(rad, shape, mode):
    def _gaussian_filter(source):
        def subscribe(observer, scheduler=None):
            def on_next(value):
                base = np.zeros(shape[:2])
                rows, cols = shape[:2]
                center = (rows/2, cols/2)
                for x in range(cols):
                    for y in range(rows):
                        if not mode:
                            base[y, x] = np.exp(-(distance((y, x),
                                                           center))**2/(2*rad**2))
                        elif mode:
                            base[y, x] = 1 - np.exp(-(distance((y, x), center))
                                                    ** 2/(2*rad**2))
                observer.on_next(value * base)

            return source.subscribe(
                on_next,
                observer.on_error,
                observer.on_completed,
                scheduler)
        return create(subscribe)
    return _gaussian_filter


def preprocess(res):
    return pipe(
        ops.map(lambda img: Image.fromarray(img)),
        ops.map(lambda img: img.resize(res)),
        ops.map(lambda img: img.convert('L')),
        ops.map(lambda img: np.flipud(np.fliplr(img))),
    )


def spectrum(mode):
    if mode:
        return pipe(
            ops.map(lambda img: np.fft.fft2(img)),
            ops.map(lambda img: np.fft.fftshift(img)),
        )
    elif not mode:
        return pipe(
            ops.map(lambda img: np.fft.fftshift(img)),
            ops.map(lambda img: np.fft.fft2(img)),
        )


rad = 20
res = (300, 300)

img = io.imread('nade1.png')
img_array = np.array(img)

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(img)
ax2 = fig.add_subplot(1, 2, 2)

src = of(img_array).pipe(preprocess(res),
                         spectrum(1),
                         gaussian_filter(rad, res, 1),
                         spectrum(0)
                         )

src.subscribe(lambda img: ax2.imshow(
    np.abs(img), cmap='gray'))

plt.show()
