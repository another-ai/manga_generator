# manga_generator
Create your manga/comic with manga_generator and Stable Diffusion XL

# Installation:
1. Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/), checking "Add Python to PATH".
2. Install [git](https://git-scm.com/download/win).
3. On terminal:
```bash
git clone https://github.com/shiroppo/manga_generator
cd manga_generator
py -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

# manga_generator.txt
File with prompts, one line = one prompt(1 scene)  
one empty line = new page  
(max 6 scenes for page)  

# env/manga_generator.env
File for configuration

# Models
If you have webui 1111 or similars you can insert your PATH in env/manga_generator.env after MAIN_DIR= and configure the other options  
or  
Download from hugginface, civitai.com or any similar website every checkpoint sd xl that you use and put them into ./models/Stable-diffusion/  
(optional) Download from hugginface, civitai.com or any similar website every LoRA sd xl that you use and put them into ./models/Lora/  
(optional) Download from hugginface, civitai.com or any similar website every VAE that you use and put them into ./models/VAE/  
Configure the env/manga_generator.env with your models

# Run:
### Method 1
Double click on ```app.bat``` on manga_generator directory
### Method 2
On terminal:
```bash
.\venv\Scripts\activate
py app.py
```
# Update:
1. ```git pull```(if error: ```git stash``` and after ```git pull```)
2. ```.\venv\Scripts\activate```
3. ```pip install -r requirements.txt```

# Versions:
- v1.0: First version
