
from turbojpeg import TurboJPEG
import imageio.v2 as imageio
import time

image_path = "/home/zhouzhiting/Data/panda_data/cano_policy_pd_2/seed_0/ep_0/step_0_cam_third.jpg" 

jpeg = TurboJPEG()
images=[]

start_time = time.time()
image = imageio.imread(image_path)
elapsed = time.time() - start_time
print(f"imageio.imread: {elapsed:.4f} s")


start_time = time.time()
with open(image_path, 'rb') as f:
    jpeg_data = f.read()
    image = jpeg.decode(jpeg_data)
elapsed = time.time() - start_time  
print(f"turbojpeg.decode: {elapsed:.4f} s")

# imageio.imread: 3.0186 s
# turbojpeg.decode: 0.0007 s