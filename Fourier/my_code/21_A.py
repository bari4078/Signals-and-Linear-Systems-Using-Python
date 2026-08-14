import numpy as np
import matplotlib.pyplot as plt


# Load and preprocess the image
image = plt.imread('noisy_image.png')  # Replace with your image file path
# show the image
plt.figure()
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.show()

if image.ndim == 3:
    image = np.mean(image, axis=2)  # Convert to grayscale

#image = image / 255.0  # Normalize to range [0, 1]
print (image.shape)
print(image.min(),image.max())
sample_rate = 1000 

denoised_image = np.zeros_like(image)

row_length = image.shape[1]
for frac in [0.02, 0.025, 0.022,0.019,0.01,0.021,0.001]:
    cutoff = max(1, int(row_length * frac))
    denoised_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        row_fft = np.fft.fft(image[i, :])
        row_fft[cutoff:-cutoff] = 0
        denoised_image[i, :] = np.fft.ifft(row_fft).real
    plt.figure()
    plt.title(f'cutoff fraction = {frac}')
    plt.imshow(np.clip(denoised_image, 0, 1), cmap='gray')
    plt.show()


plt.imsave('denoised_image.png', denoised_image, cmap='gray')


plt.figure()
plt.title('Denoised Image')
plt.imshow(denoised_image, cmap='gray')
plt.show()
