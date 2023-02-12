from transformers import CLIPProcessor, CLIPModel

def smap_grad(img):    
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return NotImplementedError