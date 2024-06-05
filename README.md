# manga_generator
Create your manga/comic with manga_generator and Stable Diffusion XL

# Installation on Windows:
1. Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/), checking "Add Python to PATH".
2. Install [git](https://git-scm.com/download/win).
3. On terminal:
```bash
git clone https://github.com/shiroppo/stable_cascade_easy
cd stable_cascade_easy
py -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```
# Installation on Linux:

1. Clone the repository:
git clone ```https://github.com/another-ai/stable_cascade_easy.git```

2. open terminal in the cloned directory: stable_cascade_easy
type the following prompt:
```python3 -m venv env```

3. to activate the virtual environment type:
```source env/bin/activate```

4. your terminal will change to (env) for the new commands. Type the following
```pip install -r requirements.txt```

5. the git pull will now work without errors. when install is finished type the following
```python3 app.py```

it will take a while to download the models and launch the Web UI in your default browser.

to launch again you can write a new file in your text editor and save in in the stable_cascade_easy directory. save the file as start.sh
here's the text you need to write in the ```start.sh``` file, you need to change "user" to your own user name:

```#!/bin/bash```

Specify the paths to your virtual environment and start.py script
```venv_path="/home/user/stable_cascade_easy"```

Open a new Gnome terminal window
```bash
gnome-terminal --working-directory=$venv_path -- bash -ic
"source env/bin/activate;
python3 app.py;
exec bash"
```

# Run on Windows:
### Method 1
Double click on ```app.bat``` on stable_cascade_easy directory
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
