Data vizualization
==================

This is a collection of tools to represent and navigate through the high-dimensional data. The algorithm t-SNE has been used to construct the 2D space so some choices and feature of the visualization may reflect that. The module should be agnostic of the data provided.

What does it needs to be executed ?
-----------------------------------

data_viz needs the following files:
* pre-processed transactions
* predictions:
    * predictor (currently only the keras NN are supported), the algo which will eat the pre-processed transactions
            it should have been trained on data ordered the same way as the raw transactions array.**or** 
    * its predictions
* 2D-projections: (optional)
    * a t-SNE (or another dimension-reduction nD-to-2D algorithm) output representing pre-processed data in a 2D-space **or**
    * parameters for t-SNE (optional, default ones are provided)
* raw transactions (optional) which will be used to display additional human-understandable info.

Usage
-----

### How to run?

Change the config file (currently the .py one).
Run the main.

### How to use ?
Navigate inside the 2D space and look at the data, selecting it in the main window (the big one). Only this one is interactive. Data is grouped by buckets ('tiles' inside 'grids' in the code), you can select buckets individually (left click) or by similarity (middle click).

Main window represents all the data in 2D space. Blue are good-predicted transactions, Red are the bad ones, Green are the special class (by default the label 0).

Below are three subplots :
* a summary of the data inside the selected buckets (see navigation)
* a heatmap of the red/blue/green representation
* a heatmap of the entropy of the buckets (normalized with the theorical max entropy of all labels)

Data viz navigation :
* left click selects a bucket of data
* middle click selects a bunch of similar buckets according to your method (#TODO MenuList to select bunching process)
* right click reset all in-memory buckets

Other options:
* show one label with the 'Show one label' field
* select all buckets containing a label with 'Select all with label'


File structures
---------------

Your files should be structured this way :
(samples are provided in the data folder, you still need to find the raw/processed datas as it too heavy to be put here.

* pre-processed transactions:
    * type: npz
    * internal structure:
        * entry x: pre-processed inputs
        * entry y_$(OUTPUT_NAME): pre-processed label to be predicted
        * (optional) entry $(OUTPUT_NAME)_encoder: humanToMachine labels labelling
    * name: $(INPUT_FILE_BASE_NAME)_x_y$(VERSION).npz
    * path: $(DATA_PATH)

* predictions:
    * type: npz
    * internal structure:
        * entry pred: the labels predicted
    * name: predictions$(VERSION).npz
    * path: $(MODEL_PATH)

* 2D-projections:
    * type: npz
    * internal structure:
        * entry x_2D: array of (float, float) datas
    * name: embedded_x_1-$(REDUCTION_SIZE_FACTOR)_$(PARAMS[0])_$(PARAMS[1]).$(PARAMS[N]).npz
    * path: $(TSNE_DATA_PATH)

* raw transactions:
    * type: npz
    * internal_structure:
        * entry originals: raw transactions
    * name: originals$(VERSION).npz
    * path: $(DATA_PATH)
    
* predictor:
    * type: keras.model
    * name $(DEFAULT_RN)$(VERSION)
    * path: $(MODEL_PATH)

Default parameters
------------------

* BASE_PATH: absolute path of where to find things
    * e.g: /home/user/data

All the following are relative path inside $(BASE_PATH)
* DATA_PATH: where are the originals data (pre-processed and raw data)
    * e.g: data/
* TSNE_DATA_PATH: where are the 2D-projected datas and models
    * e.g: tsne/
* MODEL_PATH: where are the NN keras models and/or its predictions
    * e.g: models/
* GRAPH_PATH: where to store fancy graph and pickle visualizations
    * e.g graph/

* INPUT_FILE_BASE_NAME: base name of pre-processed datas
    * e.g: processed_prediction, processed_for_meta, one_hot
* DEFAULT_RN: name of keras NN model for predictions
    * e.g: one_hot_1000-600-300-200-100_RN
* VERSION: suffix to identify a unique version of all files
    * e.g: _20170614

* e.g:
    * processed_prediction_x_y_20170614.npz
    * one_hot_1000-600-300-200-100_RN_20170614

* PARAMS: parameters of the t-SNE projections, all combinations will be loaded (or learnt)
    * e.g: {
        'perplexities'  : [30,40,50],
        'learning_rates': [500, 800, 1000],
        'inits'         : ['random', 'pca'],
         'n_iters'       : [5000, 15000]
         }
* REDUCTION_SIZE_FACTOR: which size of subset of data you want to load
    * e.g: 50 for local, 30 for ovh, 15 for epinal
    
* OUTPUT_NAME: type of label you want to load, predict, and visualize
    * e.g: account

* DO_CALCULUS: does the main should learn (if not will learn) the t-SNE representations / RN predictions
    * e.g: True
