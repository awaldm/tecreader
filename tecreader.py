import tecplot as tp
import numpy as np
import sys, os, fnmatch
#from memory_profiler import profile
import time
from functools import wraps
import multiprocessing as mp
import re as re
import numbers
'''
This module facilitates loading of Tecplot-formatted binary file series, usually from Tau results.
The approach is as follows:
- obtain a file list from the path in question using get_sorted_filelist()
- set start_i and end_i in order to get a sublist, optionally you can use get_cleaned_filelist()
- use either

1)    in_data = tecreader.read_series_parallel([plt_path + s for s in filelist], zone_no, varnames, n_workers)

or

2)    in_data = tecreader.read_series(plt_path, zone_no, varnames)
to obtain a numpy array of shape (n_points, n_timesteps, n_variables)

or

3)    cp, dataset = tecreader.get_series(plt_path, zonelist, start_i, end_i)
as a wrapper for the above steps to obtain a numpy array from a folder containing a file series

'''

def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ("Total time running %s: %s seconds" %
               (function.func_name, str(t1-t0))
               )
        return result
    return function_timer

def tec_get_dataset(filename, zone_no=None, in_vars=['X', 'Y', 'Z'], dataset_only = False):
    """
    Obtain a single Tecplot zone and its coordinates.
    Useful in a situation when further processing yields some data.
    """
    if isinstance(zone_no,numbers.Integral):#(int, long)):
        zones = [zone_no]
    else:
        zones = zone_no
    dataset = tp.data.load_tecplot(filename, zones=zone_no, variables=in_vars, read_data_option = tp.constant.ReadDataOption.Replace, collapse=True)
    # even though we loaded only a single zone, we need to remove the preceding zombie zones
    # otherwise they will get written and we have no chance to read properly (with paraview)
    # this works only (and is necessary) when using the single zone/no extra datasetfile approach
    #for i in range(zones[0]):
    #    dataset.delete_zones(0)
    if dataset_only:
        return dataset
    else:
        data = {}

        #dataset = tp.data.load_tecplot(in_path+filename, zones=zones, variables = in_vars, read_data_option = tp.constant.ReadDataOption.Replace)
        for zone in dataset.zones():
            for variable in in_vars:
                #print variable
                array = zone.values(variable)
                data[variable] = np.array(array[:]).T
        return dataset, data

def save_plt(newvar, dataset, filename, addvars = True, removevars = True):
    varnames = newvar.keys()
    #print varnames
    offset = 0

    if addvars:
        for keys, _ in newvar.items():
            dataset.add_variable(keys)

    for zone in dataset.zones():
        #print('adding variable ' + varname1 + ' to zone ' + zone.name)
        zone_points = zone.num_points
        print('zone has ' + str(zone_points) + ' points')
        #print(len(array1[offset:offset+zone_points]))
        for var in varnames:
            print(var)
            zone.values(var)[:] = newvar[var][offset:offset+zone_points].ravel() # this is what the pytecplot example uses

        offset = offset + zone_points
    tp.data.save_tecplot_plt(filename, dataset=dataset)
    if removevars:
        for keys, _ in newvar.items():
            dataset.delete_variables(dataset.variable(keys))



"""
def tec_get_dataset(filename, zone_no=None, in_vars=['X', 'Y', 'Z']):

    '''
    Obtain a single Tecplot zone and its coordinates.
    Useful in a situation when further processing yields some data.
    '''
    if isinstance(zone_no,(int, long)):
        zones = [zone_no]
    else:
        zones = zone_no
    dataset = tp.data.load_tecplot(filename, zones=zone_no, variables=in_vars, read_data_option = tp.constant.ReadDataOption.Replace)
    # even though we loaded only a single zone, we need to remove the preceding zombie zones
    # otherwise they will get written and we have no chance to read properly (with paraview)
    # this works only (and is necessary) when using the single zone/no extra datasetfile approach
    for i in range(zones[0]):
        dataset.delete_zones(0)


    #dataset = tp.data.load_tecplot(in_path+filename, zones=zones, variables = in_vars, read_data_option = tp.constant.ReadDataOption.Replace)
    for zone in dataset.zones():
        for variable in in_vars:
            print variable
            array = zone.values(variable)
            data[variable] = np.array(array[:]).T
    return dataset, data
"""

def read_series(source_path, zone, varnames, szplt=False, gridfile=None, include_geom=True, verbose=False):
    if isinstance(source_path, str):
        filelist, num_files= get_sorted_filelist(source_path, 'plt')
    else:
        filelist = source_path
        num_files = len(filelist)
    print(num_files)
    print('reading zones: ' + str(zone))
    start_time = time.time()
    f_r = []
    for i in range(len(filelist)):
       # _,_,_, data_i, dataset = load_tec_file(source_path + filelist[i], szplt=True, verbose=False, varnames=varnames, load_zones=zone, coords=False, deletezones=False, replace=True)
        _,_,_, data_i, dataset = load_tec_file(filelist[i], szplt=szplt, verbose=verbose, varnames=varnames, load_zones=zone, coords=False, deletezones=False, replace=True, gridfile=gridfile)

        f_r.append(data_i)

    f_r = np.asarray(f_r)
    #print('finished reading after ' + str(time.time()-start_time) + ' s')
    #print('shape of result before transposition: ' + str(f_r.shape))
    return np.transpose(f_r, (1,0,2))


def parallel_load_wrapper(input_arg):
    '''
    Helper function for read_series_parallel
    '''
    (filename, zone, varnames, szplt, include_geom, gridfile) = input_arg
    #print(filename)
    #print(gridfile)
    _,_,_,data_i,dataset = load_tec_file(filename, szplt=szplt, verbose=False, varnames=varnames, load_zones=zone, coords=False, deletezones=False, replace=True, include_geom=include_geom, gridfile=gridfile)
    #print(data_i.shape)
    return data_i

def read_series_parallel(source_path, zone, varnames, workers, include_geom=True, szplt=False, gridfile=None, verbose=False):
    """ Reads a file list using multiple threads
    Parameters
    ----------
    source_path : str or list
        Path containing the files or list of files
    zone : list of str or int
        Zones to load
    varnames : list of str
        Variables to load
    workers : int
        Number of parallel workers

    Raises
    ------

    Description
    -----------

    Read a time series of files in binary Tecplot format using pytecplot, tested with the PyTecplot version supplied with 2017R3.
    This can only work if there is a connection to a Tecplot licence server!
    x, y, z are always read. varname is required and should contain single quotes e.g. 'cp'
    This function does not care whether data is structured or not

    TODO: no idea whether this is any faster than the serial version. Probably
    not, needs benchmarking.
    """

    nProc = workers
    pool = mp.Pool(processes=nProc)
    if isinstance(source_path, str):
        filelist, num_files= get_sorted_filelist(source_path, 'plt')
    else:
        filelist = source_path
        num_files = len(filelist)
    #filelist = filelist[0:31]
    #num_files = len(filelist)
    print('setting up parallel reader using ' + str(workers) + ' workers')
    print('reading zones: ' + str(zone))
    #print('filelist: ' + str(filelist))
    start_time = time.time()
    #print source_path
    #print(filelist[0])
    #args = ((filelist[i], zone, varnames, szplt, include_geom) for i in range(len(filelist)))
    args = ((filelist[i], zone, varnames, szplt, include_geom, gridfile) for i in range(len(filelist)))

    #print('getting geometry: ' + str(include_geom))
    #print('using gridfile: ' + str(gridfile))
    results = pool.map_async(parallel_load_wrapper, args)
    f_r = results.get()
    f_r = np.asarray(f_r)

    pool.close()
    pool.join()

    print('finished reading after ' + str(time.time()-start_time) + ' s')
    print('shape of result before transposition: ' + str(f_r.shape))
    if f_r.ndim < 3:
        f_r = f_r[:,:,None]
    return np.transpose(f_r, (1,0,2))


def get_series(plt_path, zone_no, start_i=None, end_i=None, datasetfile=None, read_velocities=True, read_cp=False, read_vel_gradients=False, read_pressure_gradients=False, stride=10, parallel=True, include_geom=True, gridfile=None, verbose=False):
    '''
    This is the most convenient user-facing function. Use this to read a series
    of .plt files from a given folder, or a subseries delimited by start_i and
    end_i.
    This is mostly aimed at files using the TAU-derived naming scheme

    Parameters
    ----------
    plt_path : str or list
        Path containing the files or list of files
    zone_no : list of str or int
        Zones to load
    datasetfile : str
        Determine via path to a file which dataset will be returned via the
        variable dataset. Set path to a file. If None then the first element
        of the file list given by plt_path will be used.
    read_* : bool
        Booleans setting what variables to read and to output

    Returns
    -------
    u,v,w or similar : numpy array
        Arrays of shape (n_points, n_timesteps) containing the desired variables
    dataset : pytecplot dataset object
        Object containing the original dataset and x,y,z spatial coordinates.
        Useful when further data processing yields data that needs to be written
        to the original geometry.

    TODO
    ----
    The output variables need to be bundled in order to simplify data passing.
    Can be done via dicts or a data container class.
    '''

    print('\nreading series...')
    print(80*'-')

    # Set up names of the variables conforming to the TAU solver's convention
    varnames = list()
    if read_velocities:
        vel_list =['x_velocity', 'y_velocity', 'z_velocity']
        for name in vel_list:
            varnames.append(name)
    if read_cp:
        varnames.append('cp')
    if read_vel_gradients:
        gradlist = ['dudx', 'dudy', 'dudz', 'dvdx', 'dvdy', 'dvdz', 'dwdx', 'dwdy', 'dwdz']
        for name in gradlist:
            varnames.append(name)
    if read_pressure_gradients:
        gradlist = ['dpdx', 'dpdy', 'dpdz']
        for name in gradlist:
            varnames.append(name)

    print('Variable names to be loaded: ' +str(varnames))


    # Get a list of files present in the given folder and select parts of it, if necessary
    filelist, num_files = get_sorted_filelist(plt_path, 'plt', stride=stride)
    filelist = get_cleaned_filelist(filelist,start_i=start_i, end_i=end_i)

    num_files = len(filelist)

    # zone number needs to be a list
    #if isinstance(zone_no,(int, long)):
    if isinstance(zone_no,numbers.Integral):#(int, long)):

        zone_no = [zone_no]
    else:
        zone_no = zone_no
    if parallel:
        in_data = read_series_parallel([plt_path + s for s in filelist], zone_no, varnames, 3, include_geom=include_geom, gridfile=gridfile)
    else:
        in_data = read_series([plt_path + s for s in filelist], zone_no, varnames, include_geom=include_geom, gridfile=gridfile)

    out_data = dict()
    print('data shape: ' + str(in_data.shape))
    if read_velocities:
        out_data['u'] = in_data[:,:,varnames.index('x_velocity')]
        out_data['v'] = in_data[:,:,varnames.index('y_velocity')]
        out_data['w'] = in_data[:,:,varnames.index('z_velocity')]
    if read_cp:
        out_data['cp'] = in_data[:,:,varnames.index('cp')]

    if read_vel_gradients:
        out_data['dudx'] = in_data[:,:,varnames.index('dudx')]
        out_data['dudy'] = in_data[:,:,varnames.index('dudy')]
        out_data['dudz'] = in_data[:,:,varnames.index('dudz')]
        out_data['dvdx'] = in_data[:,:,varnames.index('dvdx')]
        out_data['dvdy'] = in_data[:,:,varnames.index('dvdy')]
        out_data['dvdz'] = in_data[:,:,varnames.index('dvdz')]
        out_data['dwdx'] = in_data[:,:,varnames.index('dwdx')]
        out_data['dwdy'] = in_data[:,:,varnames.index('dwdy')]
        out_data['dwdz'] = in_data[:,:,varnames.index('dwdz')]

    if read_pressure_gradients:
        out_data['dpdx'] = in_data[:,:,varnames.index('dpdx')]
        out_data['dpdy'] = in_data[:,:,varnames.index('dpdy')]
        out_data['dpdz'] = in_data[:,:,varnames.index('dpdz')]


    if gridfile is not None:
        dataset = tec_get_dataset(gridfile, zone_no = zone_no[0:(len(zone_no)/2)], dataset_only=True)
        #print('loaded dataset from gridfile, zones ' + str(zone_no[0:(len(zone_no)/2)]))
    else:
        if datasetfile is None:
            dataset = tec_get_dataset(plt_path + filelist[0], zone_no = zone_no, dataset_only=True)
        else:
            dataset = tec_get_dataset(datasetfile, dataset_only=True)

    return out_data, dataset

'''

The actual data are found at the intersection of a Zone and Variable and the resulting object is an Array. The data array can be obtained using either path:

>>>

>>> # These two lines obtain the same object "x"
>>> x = dataset.zone('My Zone').values('X')
>>> x = dataset.variable('X').values('My Zone')


'''

def get_sorted_filelist(source_path, extension, sortkey='i', stride=10):
    print('looking for data in '  + source_path + ' using pattern *.' + extension + ', sortkey = ' + sortkey)
    if os.path.isdir(source_path): # folder of files
        filelist = fnmatch.filter(os.listdir(source_path), '*.' + extension)
        print('length of unsorted file list: ' + str(len(filelist)))
        if sortkey == 'i':
            filelist = sorted(filelist, key = get_i)
        elif sortkey == 'PIV':
            filelist = sorted(filelist)
        else:
            filelist = sorted(filelist, key = get_domain)
        num_files = len(filelist)
    else: # single file
        filelist = list(source_path)
        num_files = 1
    newlist= []
    for file in filelist:
        if sortkey=='i':
            if (get_i(file) % stride) == 0:
                newlist.append(file)
        else:
            newlist.append(file)

    num_files = len(newlist)

    return newlist, num_files

def get_i(s):
    i = re.search('i=\d+',s).group()
    i = i.split('i=')[1]
    return int(i)

'''
def get_i(s):
    temp= s.split('i=')[1]
    i = temp.split('.plt')[0]
    if '_t=' in i:
        i = i.split('_t=')[0]
    else:
        i = i.split('_')[0]

    return int(i)
'''
def get_t(s):
    temp= s.split('t=')[1]
    t = temp.split('e')[0]
    return int(t)

def get_domain(s):
    temp= s.split('domain_')[1]
    dom = temp.split('.plt')[0]
    return int(dom)

def load_plt_series(plt_path, zone_no, varnames=None, parallel=0, start_i=None, end_i=None):
    '''
    call the serial or parallel series loaders
    call the file list sorting routines
    load one extra dataset
    '''
    filelist, num_files = get_sorted_filelist(plt_path, 'plt')

    filelist = get_cleaned_filelist(filelist,start_i=start_i, end_i=end_i)

    get_sorted_filelist(source_path, extension, sortkey='i', stride=10)

    if parallel > 0:
        in_data = read_series_parallel([plt_path + s for s in filelist], zone_no, varnames, parallel)
    else:
        in_data = read_series(plt_path, zone, varnames)

    return coords, data


def get_zones(dataset, keepzones='byname', zonenames=['hexa']):
    '''
    select zones by number, name, dimension or some other aspect
    '''
    if keepzones =='byname':
        # problem#
        pass
        #[Z for Z in dataset.zones() if 'Wing' in Z.name]
        #found = any(word in item for item in wordlist)
        delzones
    if keepzones == '3D':
        zones2D = [Z for Z in dataset.zones() if Z.rank < 3]
        print(str(zones2D))
        dataset.delete_zones(zones2D)
        print('deleted 2D zones, what remains is')
        print('number of zones: ' + str(dataset.num_zones))
        for zone in coords_data.zones():
            print(zone.name)
    elif keepzones == '2D':
        zones2D = [Z for Z in dataset.zones() if Z.rank > 2]
        print(str(zones2D))
        dataset.delete_zones(zones2D)
        print('deleted 3D zones, what remains is')
        print('number of zones: ' + str(dataset.num_zones))
        for zone in dataset.zones():
            print(zone.name)

#@profile(precision=4)
@fn_timer
def read_ungathered(source_path, load_zones=None, szplt=False, varnames=None, shape=None, verbose=False):
    if not os.path.isdir(source_path):
        raise IOError(str(source_path) + ' does not exist.')

    # special case: if no varnames are given, we assume that we want velocities
    if varnames is None:
        varnames=['X', 'Y', 'Z', 'x_velocity', 'y_velocity', 'z_velocity']
    if szplt:
        filelist, num_files= get_sorted_filelist(source_path, 'szplt')
    else:
        filelist, num_files= get_sorted_filelist(source_path, 'plt', sortkey='domain')

    print('found ' + str(num_files) + ' files total in the folder')

    s=filelist[0]

    readoption = tp.constant.ReadDataOption.Append

    data = np.empty((0,6))
    for file in range(len(filelist)):
        s = filelist[file]
        #dataset = tp.data.load_tecplot(source_path+s, zones=load_zones, read_data_option=readoption)
        try:
            _,_,_,data_i, _ = load_tec_file(source_path + s, szplt=False, varnames=varnames, load_zones=load_zones, coords=False, verbose=False, deletezones=False, replace=True)
            data = np.vstack((data, data_i))
        except NameError:
            print('requested zone not found, no problem')

    print(data.shape)
    print('velocity snapshot shape: ' + str(data.shape) + ', size ' + str(size_MB(data)) + ' MB')

def get_cleaned_filelist(filelist, start_i, end_i):
    startindex = None
    endindex = None
    print('looking for starting i ' + str(start_i) + ' end ending i ' + str(end_i) + ' in a file list of length ' + str(len(filelist)))
    for item in filelist:
        if get_i(item) == start_i:
            startindex = filelist.index(item)
        elif get_i(item) == end_i:
            endindex = filelist.index(item)
    print('found indices ' + str(startindex) + ' and ' + str(endindex))
    if startindex is None or endindex is None:
        return filelist
    else:
        return filelist[startindex:endindex]

def cleanlist(filelist, skip=None, start_i=None, end_i=None, num_i=None, di=10):
    ###########################################################################
    # sort out the various options for starting, skipping and ending
    # where to start? is start_i or skip given? (these are mutually exclusive)
    if skip is None:
        skip = 0

    if start_i is None:
        start_i = get_i(s)
    else:
        start_i = start_i + skip
    print('starting at I ' + str(start_i))

    if num_i is None and end_i is None:
        maxcount = num_files - (skip // di)
        print('going to the last file')
        end_i = get_i(filelist[-1])
    elif num_i is not None and end_i is None:
        end_i = start_i + (num_i-1) * di
        print('ending at I ' + str(end_i))
    elif num_i is None and end_i is not None:
        maxcount = (end_i - start_i + di) // di
        print('ending at I ' + str(end_i))
        num_i = (end_i - start_i + di) // di
    else:
        sys.exit('num_i and end cannot be both specified at the same time')
    maxcount = (end_i - start_i + di) // di
    num_i = maxcount

    return start_i, end_i, num_i


#@profile(precision=4)
def read_tec_bin_series(source_path, load_zones=None, szplt=False, varnames=None, shape=None, skip=None, start_i=None, end=None, num_i=None, di=10, verbose=False):
    """ Loads a time series of binary Tecplot data using the PyTecplot API.
    Parameters
    ----------
    source_path : str
        Path containing the data.
    load_zones : list of str or int, default None
        Lists the zones to be loaded
    szplt : bool
        sets whether data is in old Tecplot PLT binary format or in the new SZPLT
    varnames : list of str, default None
        Variables to load. If None then all are kept
    skip : int
        number of files to skip at the beginning. Mutually exclusive with start_i
    start_i : int
        First file to load. Mutually exclusive with skip.
    end : int
        Last file to load. Mutually exclusive with num_i
    num_i : int
        number of files to load. Mutually exclusive with Append
    di :
        the stride
    verbose :
        verbosity of console output

    Raises
    ------
    IOError
        If path does not exist.

    Description
    -----------

    Read a time series of files in binary Tecplot format using pytecplot, tested with the PyTecplot version supplied with 2017R3.
    This can only work if there is a connection to a Tecplot licence server!
    x, y, z are always read. varname is required and should contain single quotes e.g. 'cp'
    This function does not care whether data is structured or not
    """
    if not os.path.isdir(source_path):
        raise IOError(str(source_path) + ' does not exist.')

    # special case: if no varnames are given, we assume that we want velocities
    if varnames is None:
        varnames=['x_velocity', 'y_velocity', 'z_velocity']
    if szplt:
        filelist, num_files= get_sorted_filelist(source_path, 'szplt')
    else:
        filelist, num_files= get_sorted_filelist(source_path, 'plt')

    print('found ' + str(num_files) + ' files total in the folder')

    s=filelist[0]

    ###########################################################################
    # sort out the various options for starting, skipping and ending
    # where to start? is start_i or skip given? (these are mutually exclusive)
    if skip is None:
        skip = 0

    if start_i is None:
        start_i = get_i(s)
    else:
        start_i = start_i + skip
    print('starting at I ' + str(start_i))

    if num_i is None and end is None:
        maxcount = num_files - (skip // di)
        print('going to the last file')
        end = get_i(filelist[-1])
    elif num_i is not None and end is None:
        end = start_i + (num_i-1) * di
        print('ending at I ' + str(end))
    elif num_i is None and end is not None:
        maxcount = (end - start_i + di) // di
        print('ending at I ' + str(end))
        num_i = (end - start_i + di) // di
    else:
        sys.exit('num_i and end cannot be both specified at the same time')
    maxcount = (end - start_i + di) // di
    num_i = maxcount

    start_i, end_i, num_i = cleanlist(filelist, skip=skip, start_i=start_i, end_i=end, num_i=num_i, di=di)

    print('reading ' + str(num_i) + ' files')

    count=0
    print('num_files: ' + str(num_i))


    ###########################################################################
    data = dict()
    for file in range(len(filelist)):
        s = filelist[file]
        i = get_i(s)

        # skip before starting I
        if (i < start_i):
            continue
        verbose = True
        if verbose:
            print('processing i=' + str(i))
        #verbose = False
        # do verbosity only on first file
        print('calling load function...\n')
        if count == 0:
            verbose = verbose
            print('acquiring dataset for later output...')

            # the first frame contains the data set of the first file, usually in order to retain the data structure for future handling
            page = tp.active_page()
            frame1 = page.active_frame()
            if szplt:
                coords_data = tp.data.load_tecplot_szl(source_path + s)
            else:
                coords_data = tp.data.load_tecplot(source_path + s, zones=load_zones, variables = [0,1,2])

            frame2 = page.add_frame()
            frame2.activate()
            x,y,z,data_i, dataset = load_tec_file(source_path + s, szplt=szplt, varnames=varnames, load_zones=load_zones,coords=True, verbose=verbose, deletezones=False, replace=False)
        else:
            _,_,_,data_i, _ = load_tec_file(source_path + s, szplt=szplt, varnames=varnames, load_zones=load_zones, coords=False, verbose=verbose, deletezones=False)

        print('shape of data array after file count '+str(count)+': ' + str(data_i.shape))


        # initialize array for entire dataset
        if count < 1:
            print(str(x))
            num_points = len(data_i)
            print('num_points: ' + str(num_points))

            for var in range(len(varnames)):
                print('creating dict field for variable ' + varnames[var] + ' to hold all ' + str(num_points) + ' points and ' + str(num_i) + ' snapshots' )
                data[varnames[var]] = np.zeros([num_points, num_i])
            print('data is dict with keys ' + str(data.keys()))
            for key in data.keys():
                print('shape of ' + str(key) + ': ' + str(data[key].shape))


        else:
            pass

        # assign the data just read to the overall dataset
        for var in range(len(varnames)):
            #print(str(var))
            #print('data_i: ' + str(type(data_i)))
            #print('data_i: ' + str(data_i.shape))
            data[varnames[var]][:,count] = data_i[:,var] # for when data_i is not a dict but a simple numpy array

        count += 1

        if get_i(s) >= end:
            break
    print('finished reader loop')


    if shape is not None:
        for var in range(len(varnames)):
            data[varnames[var]] = data[varnames[var]].reshape(rows,cols,num_files)

    # reactivate the frame containing the very first data set
    # delete the frame containing the last data set (possibly unnecessary, just some cleanup)
    frame1.activate()
    page.delete_frame(frame2)

    return x, y, z, data, frame1.dataset

def tec_data_info(dataset):
    print('number of variables: ' + str(dataset.num_variables))
    print('number of zones: ' + str(dataset.num_zones))
    print('title of dataset: ' + str(dataset.title))
    for variable in dataset.variables():
        print('variable: ' +str(variable.name))
        #array = variable.values('hexa')
    for zone in dataset.zones():
        print('zone: ' + str(zone.name))
        print(zone)
        #x_array = zone.variable('X') # this does not work

def size_MB(array):
    return array.nbytes / (1024*1024)

def get_coordinates(dataset, caps = True):
    '''
    TODO
    ----
    We need to tell the function whether coordinate variables are uppercase or
    lowercase. This should be automated in some way.
    '''
    zone_no = 0
    print('getting coordinates from number of zones: ' + str(dataset.num_zones))

    for zone in dataset.zones():
        if caps:
            array = zone.values('X')
        else:
            array = zone.values('x')
        b1 = np.array(array[:]).T
        if caps:
            array = zone.values('Y')
        else:
            array = zone.values('y')

        b2 = np.array(array[:]).T
        if caps:
            array = zone.values('Z')
        else:
            array = zone.values('z')

        b3 = np.array(array[:]).T

        if zone_no == 0:
            x = b1
            y = b2
            z = b3
        else:
            x = np.hstack((b1, x))
            y = np.hstack((b2, y))
            z = np.hstack((b3, z))
        #print(str(zone_no))
        zone_no += 1

    print(str(x))
    print(str(x.shape))

    print(str(y.shape))
    print(str(z.shape))
    #sys.exit(0)
    return x,y,z



def load_tec_file(filename, szplt=False, varnames=None, load_zones=[0,1], coords=True, verbose=False, deletezones=True, replace=True, include_geom=True, gridfile=None):
    '''
    Read a Tecplot binary file using pytecplot, tested with version 0.9 (included in Tecplot 2017R3)
    and 0.12 (Tecplot 2018R2)

    This reads a single data file and concatenates the values of all zones to a numpy column.
    If more than one variable is requested then the variables are added as further columns.
    The returned dataset is a numpy array of dimensions (n_points, n_variables) where
    n_points is the total number of points in the zones

    The zones are deleted at the end of the function. The rationale for this is that Tecplot
    keeps the dataset persisent, i.e. when this function is called repeatedly then the new
    data is simply added. The delete_zones function deletes all but one zone.

    Issues
    ------

    TODO: delete_zones may be better called somewhere else


    Parameters
    ----------
    szplt : bool
        True if the SZPLZ loader should be used
    varnames : list of str
        Variable names in the Tecplot file (check via pltview command line tool)
    load_zones : list of int
        Zone numbers requested to be loaded
    coords : bool
        Obtain spatial coordinates x,y,z. This is obsolete, but may still work.
    verbose : bool
        Toggle whether more information is requested, i.e. whether this function
        should be talkative.
    deletezones : bool
        Delete all zones after reading. This is deprecated and should not be used
        (i.e., set this to False).
    replace : bool
        Load mode of the Tecplot reader function. Replaces the previously loaded
        dataset if True, appends if False.

    '''
    if gridfile is not None:
        filename = [gridfile, filename]
    if varnames is None:
        varnames=['x_velocity', 'y_velocity', 'z_velocity']
    if verbose:
        print('loading tecplot file '+str(filename) + ' and looking for variables ' + str(varnames))
    if replace:
        readoption = tp.constant.ReadDataOption.Replace
    else:
        readoption = tp.constant.ReadDataOption.Append

    # load the actual data. in case of SZPLT all zones are loaded
    if szplt:
        print('starting szplt loader')
        dataset = tp.data.load_tecplot_szl(filename, read_data_option=readoption)
        #get_zones(dataset, zones)
    else:
        if load_zones is not None:
            if not all(isinstance(x, (int)) for x in load_zones):
                raise ValueError
            if verbose:
                print('loading tec files: ' + str(filename))
            dataset = tp.data.load_tecplot(filename, zones=load_zones, read_data_option=readoption, include_geom = include_geom)
        else:
            dataset = tp.data.load_tecplot(filename, read_data_option=readoption, include_geom = include_geom)
    #print('done loading')
    if verbose:
        #tec_data_info(dataset)
        print('done with dataset info')
    # apparently tecplot appends to its dataset all the time, insteady of opening a new one
    # basically it adds the zones of each time step to the dataset, ending up with a lot of zones after a while
    # ---> use delete_zones?
    #dataset.delete_zones(dataset.zone('Zone 2'))
    # or
    #dataset.delete_zones( ([dataset.zone for z in dataset.zones()] )) or something
    # or simply dataset.delete_zones(range(8)) to delete the first 8


    #############################################
    # this accumulates all points of a single variable into a single numpy array by stacking all zones on top of each other
    # this is very useful.
    if coords is True:
        x,y,z = get_coordinates(dataset)
        if verbose:
            print('obtained coordinates, length: ' + str(len(x)))
    else:
        x=None
        y=None
        z=None

    count = 0
    zone_no = 0
    #n_vars = len(varnames)


    # we need a way to build a list of relevant zones, since we cannot delete e.g. the very first zone
    #load_zones = ['flunten', 'floben']
    # this is probably super unnecessary by now, should test removing everything between if and else
    if load_zones is not None:
        if verbose:
            print('zones to load: ' + str(load_zones))
        if any(isinstance(x, (str)) for x in load_zones):
            # load_zones is a list of strings
            zonelist = [Z for Z in dataset.zones() if Z.name in load_zones]
        elif any(isinstance(x, (int)) for x in load_zones):
            pass
            zonelist = [Z for Z in dataset.zones()]
            #zonelist = [dataset.zone(ind) for ind in load_zones]
    else:
        zonelist = [Z for Z in dataset.zones()]

    #nz = dataset.num_zones
    nz = len(zonelist)
    if verbose:
        print('number of zones: ' + str(nz))

    if nz < 1:
        raise NameError
    # read zones and do what needs to be done (convert arrays to numpy)
    # the idea is that, at the end, data is an array of shape (n_points, n_vars)
    for zone in zonelist:
        if verbose:
            print('handling zone: ' + str(zone.name) + ', rank: ' + str(zone.rank) + ' with ' + str(len(varnames)) + ' variables')

        # each variable gets a column
        for var in range(len(varnames)):
            #print(varnames[var])
            array = zone.values(varnames[var])
            if var == 0:
                b = np.array(array[:])
            else:
                b = np.vstack((b, np.array(array[:])))
            if verbose:
                print('shape of current array b for zone ' + str(zone.name) + ' , before transpose: ' + str(b.shape))
        # now we have an array b of shape (n_vars, n_points_in_this_zone)
        # transpose this
        # if there is only one variable, a simple numpy transposition does not work. in that case, we need to specify the dimension/shape
        if len(varnames) == 1:
            b.shape = (1, len(b))
            print('only 1 variable, shape of b is: ' + str(b.shape))
        b = b.T
        if zone_no == 0: # get first zone
            #data = np.zeros((len(b), 1))
            data = b
            if verbose:
                print('loaded first zone, shape is: ' + str(data.shape))
        else:
            if verbose:
                print('before stacking data shape: ' + str(data.shape))
            data = np.vstack((data, b))
            if verbose:
                print('after stacking: ' + str(data.shape))
        count = count + 1
        zone_no = zone_no + 1
    if verbose:
        print('velocity snapshot shape: ' + str(data.shape) + ', size ' + str(size_MB(data)) + ' MB')

    # This is probably not necessary anymore
    if deletezones:
        try:
            for zone in dataset.zones():
                dataset.delete_zones(zone)
                #dataset.delete_zones(dataset.zones())
        except (tp.exception.TecplotLogicError):
            print('normal exception when deleting, no worries')
    if verbose:
        print('\nload function done \n\n')
        print(data.shape)

    if include_geom is False:
        x= None
        y=None
        z=None
    return x,y,z,data, dataset



