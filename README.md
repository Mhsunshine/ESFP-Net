### Event-guided Body Structure and Prototype Learning for Clothes-changing Person Re-identification

#### Requirements
- Python 3.10
- Pytorch 2.0.1+cu118
- apex

#### About Models 
- The main model is implemented in the Python file named ESFPNET within the models folder, while the PFSP module can be found in the Python file named PFSP.

#### About Dataset
- Both the RGB and event versions of the CCVID dataset will be uploaded to Baidu Netdisk soon, accompanied by the V2E conversion script for event modality generation.
- 
#### Get Started
- Replace `_C.DATA.ROOT` and `_C.OUTPUT` in `configs/default_img.py&default_vid.py`with your own `data path` and `output path`, respectively.
- Run `script.sh`




