# Information-Retrival-System

## Overview
This project involves Information-Retrival-System using various libraries and algorithms. The main goal is to implement and analyze Information-Retrival techniques on the Cranfield dataset.

## Requirements
To run this project, you need to have the following libraries installed:

1. `scipy` (version 1.12.0)
2. Microsoft C++ Build Tools (version >= 14.0)
3. `nltk` (version 3.8.1)
4. `Punkt`
5. `gensim` (version 4.3.2)
6. `scikit-learn` (version 1.4.2)
7. `regex`
8. Python 3.12
9. `matplotlib`
10. `argparse`
11. `sys`

### Important Note
If the installed version of any package does not match the specified version, you may encounter errors. 

### Model Download
When running the project for the first time, the Word2Vec model will be downloaded, which is approximately 1.7 GB. This is a one-time download, and future runs will use the cached model.<br>







<br>
<br>


### Make sure all the python files and dataset folder and output folder are in same folder and run this command from this common directory only here i have taken template_code_part2 as common directory and all python files and cranfield dataset folder and output folder in template_code_part2 directory only.


## Project Structure
Ensure that all the Python files, dataset folder, and output folder are in the same directory. The example directory is `template_code_part2`, which should contain:
- All Python files
- The Cranfield dataset folder
- The output folder


## Running the Project

if you want you can replace this punkt and ptb by naive to run code which is implemented manually and not inbuilt function used

### Command for manual query run : with punkt and ptb:<br>

Just change dataset path and output path

```bash
python main.py -custom -dataset "D:/Desktop/Jupyter_codes/template_code_part2/cranfield/" -out_folder "D:/Desktop/Jupyter_codes/template_code_part2/output/" -segmenter punkt -tokenizer ptb

```


### Command for cranfield dataset queries run :

```bash
python main.py -dataset "D:/Desktop/Jupyter_codes/template_code_part2/cranfield/" -out_folder "D:/Desktop/Jupyter_codes/template_code_part2/output/" -segmenter punkt -tokenizer ptb
```


You can see plots in outfut folder 
