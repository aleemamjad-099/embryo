from ultralytics import YOLO
import cv2

# 1. Model load karein (Yaqeen karlein ke best.pt isi folder mein ho)
model = YOLO('best.pt')

# 2. Image ka path dein jo aapne test karni hai
image_path = '20230304-17097-3-F15-157.jpg' 

# 3. Predict karein
results = model.predict(source=image_path, conf=0.5)

# 4. Result dikhayein
for r in results:
    im_array = r.plot()  # Boxes banayein
    cv2.imshow('Embryo Detection Result', im_array)
    cv2.waitKey(0)  # Koi bhi key dabane par window band hogi
    cv2.destroyAllWindows()