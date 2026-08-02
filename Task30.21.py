import cv2
import numpy as np
import matplotlib.pyplot as plt

Crop_Start = 0.2
Crop_End = 0.8
W_Resize = 320
Rotation_Angle = 15
Blend_Weight_Rotated = 0.5
Blend_Weight_Flipped = 0.5
Mask_Radius = 90

img = cv2.imread(r"C:\Users\Etijah\Pictures\Screenshot_20260722_232311_ChatGPT.jpg")

if img is None:
    raise FileNotFoundError("The Image not Loaded")

H, W = img.shape[:2]
print(f"The Shape of Original Image : {img.shape}")

img_Crop = img[
    int(Crop_Start * H):int(Crop_End * H),
    int(Crop_Start * W):int(Crop_End * W)
]
print(f"The Shape of Cropped Image : {img_Crop.shape}")

H_Crop, W_Crop = img_Crop.shape[:2]
H_Resize = int((H_Crop / W_Crop) * W_Resize)

img_Resized = cv2.resize(
    img_Crop,
    (W_Resize, H_Resize),
    interpolation=cv2.INTER_AREA
)
print(f"The Shape of Resized Image : {img_Resized.shape}")

Center_X = W_Resize // 2
Center_Y = H_Resize // 2

M = cv2.getRotationMatrix2D(
    (Center_X, Center_Y),
    Rotation_Angle,
    1
)

img_Rotated = cv2.warpAffine(
    img_Resized,
    M,
    (W_Resize, H_Resize),
    borderMode=cv2.BORDER_REPLICATE
)
print(f"The Shape of Rotated Image : {img_Rotated.shape}")

img_Flipped = cv2.flip(img_Rotated, 1)
print(f"The Shape of Flipped Image : {img_Flipped.shape}")

img_Blended = cv2.addWeighted(
    img_Rotated,
    Blend_Weight_Rotated,
    img_Flipped,
    Blend_Weight_Flipped,
    0
)
print(f"The Shape of Blended Image : {img_Blended.shape}")

H_Blend, W_Blend = img_Blended.shape[:2]

Center_X = W_Blend // 2
Center_Y = H_Blend // 2

Y, X = np.ogrid[:H_Blend, :W_Blend]

Circle = (
    (X - Center_X) ** 2 +
    (Y - Center_Y) ** 2
) <= Mask_Radius ** 2

mask = np.zeros((H_Blend, W_Blend), dtype=np.uint8)
mask[Circle] = 255

img_Masked = cv2.bitwise_and(
    img_Blended,
    img_Blended,
    mask=mask
)
print(f"The Shape of Masked Image : {img_Masked.shape}")

img_Masked_Gray = cv2.cvtColor(
    img_Masked,
    cv2.COLOR_BGR2GRAY
)
print(f"The Shape of Masked Gray Image : {img_Masked_Gray.shape}")

fig, axes = plt.subplots(1, 7, figsize=(22, 5))

axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original")

axes[1].imshow(cv2.cvtColor(img_Crop, cv2.COLOR_BGR2RGB))
axes[1].set_title("Cropped")

axes[2].imshow(cv2.cvtColor(img_Resized, cv2.COLOR_BGR2RGB))
axes[2].set_title("Resized")

axes[3].imshow(cv2.cvtColor(img_Rotated, cv2.COLOR_BGR2RGB))
axes[3].set_title("Rotated")

axes[4].imshow(cv2.cvtColor(img_Flipped, cv2.COLOR_BGR2RGB))
axes[4].set_title("Flipped")

axes[5].imshow(cv2.cvtColor(img_Masked, cv2.COLOR_BGR2RGB))
axes[5].set_title("Masked")

axes[6].imshow(img_Masked_Gray, cmap="gray")
axes[6].set_title("Grayscale")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig(r"C:\Users\Etijah\Desktop\CVImage\img.jpg")
plt.show()













