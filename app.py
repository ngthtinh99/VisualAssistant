# Import libraries
import torch
import gradio as gr

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from models.blip import blip_decoder
from models.blip_vqa import blip_vqa


# Use GPU if it's available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Download Image Captioning Model
image_size = 384
transform = transforms.Compose([
    transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]) 

model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth'
    
model = blip_decoder(pretrained=model_url, image_size=384, vit='large')
model.eval()
model = model.to(device)


# Download Visual Question Answering Model
image_size_vq = 480
transform_vq = transforms.Compose([
    transforms.Resize((image_size_vq,image_size_vq),interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]) 

model_url_vq = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_vqa.pth'
    
model_vq = blip_vqa(pretrained=model_url_vq, image_size=480, vit='base')
model_vq.eval()
model_vq = model_vq.to(device)


# Inference funtion
def inference(raw_image, question):
    if not question:
        image = transform(raw_image).unsqueeze(0).to(device)   
        with torch.no_grad():
            caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
            # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
            return caption[0]
    else:   
        image_vq = transform_vq(raw_image).unsqueeze(0).to(device)  
        with torch.no_grad():
            answer = model_vq(image_vq, question, train=False, inference='generate') 
        return answer[0]


# Deploy
title       = 'Visual Assistant'

description = '''<p style='text-align: center'>
                 Take or upload a photo,
                 enter a question (optional)
                 and click Submit
                 to get information from the photo.
                 </p>'''

article     = '''<p style='text-align: center'>
                 <a href='https://www.accessibilitydesigncompetition.com/'>A product of Visual US Team at ADC 2023</a>
                 </p>'''

inputs      = [gr.inputs.Image(type='pil'), gr.inputs.Textbox(label='Question')]
outputs     = gr.outputs.Textbox(label='Output')

gr.Interface(inference, inputs, outputs, title=title, description=description, article=article).launch(enable_queue=True, share='True')
