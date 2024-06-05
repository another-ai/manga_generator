# manga_generator
Create your manga/comic with manga_generator and Stable Diffusion XL

# Installation on Windows:
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
# Installation on Linux:

1. Clone the repository:
git clone ```https://github.com/another-ai/manga_generator.git```

2. open terminal in the cloned directory: manga_generator
type the following prompt:
```python3 -m venv venv```

3. to activate the virtual environment type:
```source venv/bin/activate```

4. your terminal will change to (venv) for the new commands. Type the following
```pip install -r requirements.txt```

5. the git pull will now work without errors. when install is finished type the following
```python3 app.py```

it will take a while to download the models and launch the Web UI in your default browser.

to launch again you can write a new file in your text editor and save in in the manga_generator directory. save the file as start.sh
here's the text you need to write in the ```start.sh``` file, you need to change "user" to your own user name:

```#!/bin/bash```

Specify the paths to your virtual environment and start.py script
```venv_path="/home/user/manga_generator"```

Open a new Gnome terminal window
```bash
gnome-terminal --working-directory=$venv_path -- bash -ic
"source venv/bin/activate;
python3 app.py;
exec bash"
```

# manga_generator.txt
File with prompts, one line = one prompt(1 scene)
one empty line = new page
(max 6 scenes for page)

# env/mange_generator.env
File for configuration

# Run on Windows:
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
