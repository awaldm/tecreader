# tecreader
python reader for binary Tecplot data using pytecplot

## Introduction
This reader wraps Tecplot's pytecplot API functionality in order to handle time series of PLT files. I wrote it for the purpose of reading long series of unsteady flow
solutions and assemling them to numpy matrices for further processing.

The module includes provisions for parallel reading via multiprocessing, and is geared very much for files produces by the TAU flow solver - i.e. if you have a different 
file naming scheme you'd need to adapt some things.


## Usage
The simplest use case is as follows:

```
u, v, w, dataset = get_series(plt_path, zone_no)
```
  * `plt_path` is the input path containing the raw data time series in Tecplot binary format
  * `zone_no` is an integer or a list of integers denoting the zones to be loaded

  * `u,v,w` are the velocity components represented as matrix of shape (points, samples)
  * `dataset` is a Tecplot dataset object 

There are various utility functions, such as for saving results (`save_plt`) or obtaining coordinates (`get_coordinates`)


## Requirements
* numpy (tested with 1.12.0)
* pandas (tested with 0.19.1)
* pytecplot (tested with 0.11.0)

## Known issues
