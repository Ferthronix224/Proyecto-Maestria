from skimage import io, transform, img_as_ubyte

for i in range(1, 293):
    img = io.imread(f'../img/originals/{i}.jpg')
    rotation = transform.rotate(img, 270)
    io.imsave(f'../img/rotated/270/{i}.jpg', img_as_ubyte(rotation))