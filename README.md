# TLC-GNN

code for paper [Link Prediction with Persistent Homology: An Interactive View](https://arxiv.org/abs/2102.10255) (ICML2021)



## requirements

Python version is 3.7, and the versions of needed packages are listed in requirements.txt



## Run experiments

```
python pipelines.py
```

to run the experiments for PubMed, Photo and Computers datasets, the results will be stored in ./scores.

If you want to run experiments for PPI datasets, you can comment out line 56 in pipelines.py.



## Setup Cython

```
cd ./sg2dgm
python setup_PI.py build_ext --inplace
```

to setup ./sg2dgm/persistenceImager.pyx

If the command does not work, a substitute solution is to copy the code in ./sg2dgm/persistenceImager.pyx to a new file named ./sg2dgm/persistenceImager.py, this might also work.



## Poster

![poster](poster.png)