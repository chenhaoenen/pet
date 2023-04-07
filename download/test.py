from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import requests 
from PIL import Image
import log
import torch

logger = log.get_logger('root')
processor = TrOCRProcessor.from_pretrained("./trocr-base-handwritten")
logger.info(f"processor load ======")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = VisionEncoderDecoderModel.from_pretrained("./trocr-base-handwritten")
model.to(device)
logger.info(f"model load ======")
# load image from the IAM dataset
url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
url = "https://img-blog.csdnimg.cn/2019011113425141.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2OTQwODA2,size_16,color_FFFFFF,t_70"

url = "https://img-blog.csdnimg.cn/20181110010102885.jpg#pic_center"
url = "https://img1.baidu.com/it/u=2301485082,293592283&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=601"

url = "https://img2018.cnblogs.com/i-beta/1029680/201911/1029680-20191110213834658-1263638515.png"
url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSoolxi9yWGAT5SLZShv8vVd0bz47UWRzQC19fDTeE8GmGv_Rn-PCF1pP1rrUx8kOjA4gg&usqp=CAU"
url = "https://pic.rmb.bdstatic.com/bjh/news/641e399460b19fe01d916c1691dd8a26.jpeg"
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
image = Image.open("a01-122-02.jpg").convert("RGB")

logger.info(f"image load ======")

pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
logger.info(f"processor finished======")

generated_ids = model.generate(pixel_values).to('cpu')

logger.info(f"model finished======")
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
logger.info(generated_text)
