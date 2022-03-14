import pptx
from pptx import Presentation
from pptx.util import Inches, Length, Pt
from os.path import join
from tqdm import tqdm

def view_layout_params(pptx_path, slides_num=1):
    pprs = Presentation(pptx_path)
    for shape in pprs.slides[slides_num].shapes:
        print(type(shape))
        pos_dict = {"height" : Length(shape.height).inches,
                    "width" : Length(shape.width).inches,
                    "top" : Length(shape.top).inches,
                    "left" : Length(shape.left).inches,}
        print(pos_dict)
        if hasattr(shape,"text"):
            print("Text ", shape.text)

        if isinstance(shape, pptx.shapes.picture.Picture):
            crop_dict={"crop_right" : shape.crop_right,
            "crop_left" : shape.crop_left,
            "crop_top" : shape.crop_top,
            "crop_bottom" : shape.crop_bottom, }
            print(crop_dict)
    return pprs
