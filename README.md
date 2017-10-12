Data vizualization
==================

This is a collection of tools to represent and navigate through the high-dimensional data. The algorithm t-SNE is default to construct the 2D space. The module should be agnostic of the data provided. It ships with MNIST.

Usage
-----
### How to install ?
```sh
$ pip install vizuka
```
or clone the repo :)

### How to run?

```sh
$ vizuka
# For a quick working example run :
$ vizuka --mnist
# Similar to copy your data and run "vizuka --image:images --version _MNIST_example"
```
You can add human-readable data visualization in data/set/RAW\_NAME+VERSION.npz (default 'originals\_MNIST\_example.npz') :

```sh
$ vizuka -s price:logdensity -s name:wordcloud
# --feature-to-show raw_variable_name:{wordcloud|counter|density|logdensity|images}
```

It assumes you already have your 2D data, projection will be done if launched for the first time (not for MNIST toy example)
You can force for PCA reduction prior to t-SNE :
```sh
$ vizuka --reduce --use_pca 0.99 # Use PCA to reduce dimension and keep 99% of explained variance, then tSNE
```

It will search in \_\_package\_\_/data/ the datas but you can force your own with __--path__ argument

* Note that if you are effectively doing big data you should uncomment MulticoreTSNE in vizuka/dim_reduction.py unless you want to discover t-SNE crashed with a segfault. Instructions for installation can be found in requirements/requirements.apt

What will I get ?
-----------------

Working examples : draw clusters, find details about inside distribution and zoom in:
![alt zoomview](docs/zoom_view.png)

Here is the view you get when you launch it :
![alt mainview](docs/main_view.png)

And if you specify a set of non-preprocessed inputs to associate with your training data you can also view them in details in a per-cluster view :

![alt clusterview](docs/cluster_view.png)


### How to use ?
Navigate inside the 2D space and look at the data, selecting it in the main window (the big one). Data is grouped by cluster, you can select cluster individually (left click).

Main window represents all the data in 2D space. Blue are good-predicted transactions, Red are the bad ones, Green are the special class (by default the label 0).

Below are three subplots :
* a summary of the data inside the selected buckets (see navigation)
* a heatmap of the red/blue/green representation
* a heatmap of the cross-entropy of each bucket empirical distribution with empirical global empirical distribution.

Data viz navigation :
* left click selects a bucket of data
* right click reset all in-memory buckets

Other options:
* filter by predictions or by real class.
* detect mouse event : if unchecked, cluster will not be selected on click (useful for zooming)
* clusterize with an algo, Dummy is a simple grid, KMeans should be used, DBSCAN is experimental.
* export x : export the raw inputs you selected in an output.csv 
* cluster borders : draw borders between clusters based on bhattacharyya similarity measure, or just all
* force number of clusters (for kmeans essentially)
* choose a different set of predictions to display

What does it needs to be executed ?
-----------------------------------

vizuka needs the following files to be put in \_\_package\_\_/data:
* pre-processed transactions with the output you want to predict.
* 2D-projections: (optional)
    * a t-SNE (or another dimension-reduction nD-to-2D algorithm) output representing pre-processed data in a 2D-space **or**
    * parameters for t-SNE (optional, default ones are provided)
* your predictions (optional but recommended)
* raw transactions (optional) which will be used to display additional human-understandable info.


Ok cool I have all the data, I also installed everything, I want to do machine learning stuff, now what ?
-----------------------------------
But all your stuff in \_\_package\_\_/data (or anywhere, and specify with __--data__ argument)
Respect this formatting :


* pre-processed transactions:
    * type: npz
    * keys:
        * entry x: pre-processed inputs
        * entry y_$(OUTPUT_NAME): pre-processed label to be predicted
        * (optional) entry $(OUTPUT_NAME)_encoder: humanToMachine labels labelling
    * name: $(INPUT_FILE_BASE_NAME)_x_y$(VERSION).npz
    * path: $(DATA_PATH)
    * ex: data/set/preprocessed_x_y_20170825.npz)

* 2D-projections: (optional)
    * type: npz
    * keys:
        * x_2D: array of (float, float) datas
    * name: embedded_x_1-$(REDUCTION_SIZE_FACTOR)_$(PARAMS[0])_$(PARAMS[1]).$(PARAMS[N]).npz
    * path: $(REDUCTED_DATA_PATH)
    * ex: data/reduced/embedded_1-1_50_10000_20170825.npz
    
* raw transactions: (optional)
    * type: npz
    * keys:
        * originals: raw transactions
	* columns: collections of string to explicit nature of the data (human-readable)
    * name: originals$(VERSION).npz
    * path: $(DATA_PATH)
    * ex: originals_20150825.npz
    
* predictions: (optional but highy recommended)
    * type: npz
    * keys:
        * pred: predictions
    * pred: $(PREDICTOR)$(VERSION)
    * path: $(MODEL_PATH)
    * ex: metapredict_20170825.npz

Typical use-case :
------------------

You have your preprocessed data ? Cool, this is the only mandatory file you need. Place it in the folder *data/set/preprocessed_inputs_VERSION.npz*, VERSION being a string specific to this specific dataset. It must contains at least the key 'x' representing the vectors you learn from. If you have both the correct output and your own predicitons (inside *data/models/ALGONAMEpredict_VERSION.npz* and key 'pred')that your algo try to predict, place it under the key 'y', the data viz will be much more useful !

Optionally you can add an *original_VERSION.npz* file containing raw data non-preprocessed. The vector should be the key "originals" and the name of the human-readable "features" in the key "columns".

Now you may want to launch Vizuka ! First do specify the parameters fitting your needs in config.py. And take some coffee. Or two. Or three. Vizuka is busy reducing the dimension.

...

Congratulations ! Now you may want to display your 2D-data, as your arble to browse your embedded space. Maybe you want to look for a specific cluster. Explore the data with graph options, zoom in and zoom out, and use the filters provided to find an interesting area.

When you are satisfied, enable "detect mouse event" to be able to select clusters. This is quite unefficient you will select smal rectangular tiles one by one, you may want to *Clusterize* using KMeans or DBSCAN.

Great now you can select whole clusters of data at once. But what's in there ? Click on the *export* button to find out in a nicely formatted csv (assuming you provided "raw" data).

You finished your session but still want to dive in the clusters later ? Just select *Save clusterization* to save your session.


Default parameters
------------------

See config.py
