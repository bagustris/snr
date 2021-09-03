# A repository for paper *" Automatic Naturalness Recognition from Acted Speech Using Neural Networks"*

This paper is accepted in APSIPA-ASC 2021, Tokyo. Pre-print and slide will be added in this repository.

by
Bagus Tris Atmaja, Akira Sasou, Masato Akagi

Email: bagus@ep.its.ac.id

> This is a template for papers that use Python codes to
> generate their results (though it can be adapted to use other technologies).
> The text is written in LaTex and tasks are generated using `pdflatex` command.
> Ideally, all results, figures and the final paper PDF should be generated by
> running a single this command in the `latex` of this repository.


> This paper compares emotional song and speech from RAVDESS dataset.
> We evaluates different features sets, feature types (region of analysis, LLD vs. HSF), 
> and classifiers for speech and song data.

![](manuscript/figures/hawaii-trend.png)

*Caption for the example figure with the main results.*


## Abstract

> In this paper, we argue that singing voice (song) is
> more emotional than speech. We evaluate different features sets,
> feature types, and classifiers on both song and speech emotion
> recognition. Three feature sets: GeMAPS, pyAudioAnalysis, and
> LibROSA; two feature types, low-level descriptors and high-level
> statistical functions; and four classifiers: multilayer perceptron,
> LSTM, GRU, and convolution neural networks; are examined on
> both songand speech data with the same parameter values. The
> results show no remarkable difference between song and speech
> data on using the same method. Comparisons of two results
> reveal that song is more emotional than speech. In addition,
> high-level statistical functions of acoustic features gained higher
> performance than low-level descriptors in this classification task.
> This result strengthens the previous finding on the regression
> task which reported the advantage use of high-level features.


## Software implementation

> Briefly describe the software that was written to produce the results of this
> paper.

All source code used to generate the results and figures in the paper are in
the `code` folder.
The calculations and figure generation are all run inside
[Jupyter notebooks](http://jupyter.org/).
The data used in this study is provided in `data` and the sources for the
manuscript text and figures are in `manuscript`.
Results generated by the code are saved in `results`.
See the `README.md` files in each directory for a full description.


## Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/bagustris/paper_template.git

or [download a zip archive](https://github.com/bagustris/paper_template).

A copy of the repository is also archived at *insert DOI here*


## Dependencies

You'll need a working Python environment to run the code.
The recommended way to set up your environment is through the
[Anaconda Python distribution](https://www.anaconda.com/download/) which
provides the `conda` package manager.
Anaconda can be installed in your user directory and does not interfere with
the system Python installation.
The required dependencies are specified in the file `requirements.txt`.

We use `pip` virtual environments to manage the project dependencies in
isolation.
Thus, you can install our dependencies without causing conflicts with your
setup (even with different Python versions).

Run the following command in the repository folder (where `environment.yml`
is located) to create a separate environment and install all required
dependencies in it:

    pip3.6 venv REPO_NAME


## Reproducing the results

Before running any code you must activate the conda environment:

    source activate REPO_NAME

To reproduce result in , run the following in order:  
```bash
```


## License

All source code is made available under a BSD 3-clause license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE.md` for the full license text.

The manuscript text is not open source. The authors reserve the rights to the
article content, which is currently submitted for publication in the
JOURNAL NAME.


## Citation
Please cite this work as:  
``` 
B.T. Atmaja and M. Akagi, “A simple repository for reproducible research based on Python recipe"
```
