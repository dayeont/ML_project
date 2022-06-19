# ML_Project_2020
Machine Learning Project 2020, Group 16: Vladimir Dmitriev, Alexander Shumilov, Veronika Zorina, Daria Gazizova, Elizaveta Noskova.

## "On scalable and efficient computation oflarge scale optimal transport"

Optimal transport is one of the popular methodsused in ML, e.g.  for domain adaptation or sam-ple generation. But due to the heavy computing load, this method is difficult to use everywhere. To solve this problem, the SPOT algorithm wasdescribed, the operation of which was tested inthis project. SPOT (Scalable Push-forward of Optimal Transport) is an implicit generative learning-based framework by which the optimal transport problem is casted into a minimax problem andthen is efficiently solved by primal dual stochastic gradient-type algorithms. In this work SPOT wastested on Gaussian distribution toy dataset andused for domain adaptation and image generation.

### Content:

1) gauss.py - SPOT algorithm realization on toy datasets (first experiment with Gaussian distributions)
2) dist.py - SPOT algorithm realization on toy datasets (second experiment with more complicated distributions)
3) spot.py - SPOT algorithm realization on MNIST dataset (was ran on cluster)
4) SPOT.ipynb - notebook with spot realization and some visualizartion (Attention! To run this file properly you need to download datasets contained in Datasets.zip from the link below and put '.ipynb' file in this folder!)
5) DASPOT.py - DASPOT algorithm realization 
6) pu.sh - example of input file to run experiments on cluster
7) Requirements.txt - python packages that required for results recreation
8) Experiment_DASPOT - folder with some results for DASPOT that did not make it to the final version of report

Full SPOT & DASPOT experiments' results are too big for github, so we decided to upload zip archive here -  https://drive.google.com/file/d/1FnBa1Dtf8oDaUXrUXya3n1KHm8zaoNc_/view?usp=sharing. 
Datasets for file SPOT.ipynb can be downloaded from here - https://drive.google.com/file/d/18dEU2wH6VBLxXzvJShwPt0ePyfTsSKHb/view?usp=sharing.
Overleaf report: https://www.overleaf.com/project/5e88efb5e03b600001260247

Please, contact alexander.shumilov@skoltech.ru if links will brake or expire.
