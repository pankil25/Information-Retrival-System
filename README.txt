You need to have following libraries to be installed to run project:

(1) scipy (version=1.12.0)
(2) Microsft c++ build tool (version>= 14.0)
(3) nltk (version=3.8.1)
(4) Punkt
(5) gensim (version=4.3.2)
(6) scikit-learn (version=1.4.2)
(7) regex
(8) python 3.12
(9) matplotlib
(10) argparse
(11) sys

If mention version is not matting installed version of packages then you might get errors

While running first time word2vec will download model which is size of around 1.7 Gb so it one time download for future it will use from cache.








Make sure all the python files and dataset folder and output folder are in same folder and run this command from this common directory only here i have taken template_code_part2 as common directory and all python files and cranfield dataset folder and output folder in template_code_part2 directory only.





if you want you can replace this punkt and ptb by naive to run code which is implemented manually and not inbuilt function used

Command for manual query run : with punkt and ptb

Just change dataset path and output path

python main.py -custom -dataset "D:/Desktop/Jupyter_codes/template_code_part2/cranfield/" -out_folder "D:/Desktop/Jupyter_codes/template_code_part2/output/" -segmenter punkt -tokenizer ptb




Command for cranfield dataset queries run :

python main.py -dataset "D:/Desktop/Jupyter_codes/template_code_part2/cranfield/" -out_folder "D:/Desktop/Jupyter_codes/template_code_part2/output/" -segmenter punkt -tokenizer ptb


You can see plots in outfut folder 