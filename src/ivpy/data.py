import pandas as pd
import numpy as np
import copy
from PIL import Image
from six import string_types

ATTACHED_DATAFRAME = None
ATTACHED_PATHCOL = None

def _typecheck(**kwargs):
    """
    This function is called before plotting, as a way to control the error
    messages users receive when they pass invalid arguments. All of the 'col'
    arguments can be None; many other arguments cannot.
    """

    """data table columns"""
    pathcol = kwargs.get('pathcol')
    xcol = kwargs.get('xcol')
    ycol = kwargs.get('ycol')
    facetcol = kwargs.get('facetcol')
    notecol = kwargs.get('notecol')

    """type checking"""
    if pathcol is not None:
        if not isinstance(pathcol,(int,pd.Series)):
            raise TypeError("""'pathcol' must be an integer or
                               a pandas Series""")
    if xcol is not None:
        if not isinstance(xcol,(string_types,int,pd.Series)):
            raise TypeError("'xcol' must be a string, int, or a pandas Series")
    if ycol is not None:
        if not isinstance(ycol,(string_types,int,pd.Series)):
            raise TypeError("'ycol' must be a string, int, or a pandas Series")
    if facetcol is not None:
        if not isinstance(facetcol,(string_types,int,pd.Series)):
            raise TypeError("""'facetcol' must be a string, int, or
                               a pandas Series""")
    if notecol is not None:
        if not isinstance(notecol,(string_types,int,pd.Series)):
            raise TypeError("""'notecol' must be a string, int, or
                               a pandas Series""")

    """
    plot settings: since None is often not allowable, we default to some
    allowable value instead of to None. This allows us to prevent user-passed
    None without raising error on default None.
    """
    thumb = kwargs.get('thumb',100)
    sample = kwargs.get('sample',100)
    idx = kwargs.get('idx',False)
    bg = kwargs.get('bg')
    ascending = kwargs.get('ascending',False)
    shape = kwargs.get('shape','square')
    ncols = kwargs.get('ncols',2)
    rounding = kwargs.get('rounding','up')
    bins = kwargs.get('bins',35)
    coordinates = kwargs.get('coordinates','cartesian')
    side = kwargs.get('side',500)
    xdomain = kwargs.get('xdomain') # needs to be None sometimes
    ydomain = kwargs.get('ydomain') # needs to be None sometimes
    xbins = kwargs.get('xbins') # needs to be None sometimes
    ybins = kwargs.get('ybins') # needs to be None sometimes
    savedir = kwargs.get('savedir','foo') # any string
    feature = kwargs.get('feature','brightness')
    aggregate = kwargs.get('aggregate',True)
    scale = kwargs.get('scale',True)
    outline = kwargs.get('outline',False)
    X = kwargs.get('X',pd.DataFrame())
    method = kwargs.get('method','kmeans')
    k = kwargs.get('k',10)
    i = kwargs.get('i',0)
    centroids = kwargs.get('centroids') # will be None sometimes

    """type checking"""
    if thumb!=False: # can only be false in show()
        if not isinstance(thumb,int):
            raise TypeError("'thumb' must be an integer")
    if not isinstance(sample,int):
        raise TypeError("'sample' must be an integer")
    elif sample==True: # necessary bc True counts as int for some reason
        raise TypeError("'sample' must be an integer")
    if not isinstance(idx,bool):
        raise TypeError("'idx' must be True or False")
    if bg is not None:
        if not isinstance(bg,(tuple,string_types)):
            raise TypeError("'bg' must be an RGB triplet or a string")
    if not isinstance(ascending,bool):
        raise TypeError("'ascending' must be True or False")
    if not any([shape=='square',shape=='circle']):
        raise ValueError("'shape' must be 'circle' or 'square'")
    if not isinstance(ncols,int):
        raise TypeError("'ncols' must be an integer")
    if not any([rounding=='up',rounding=='down']):
        raise ValueError("'rounding' must be 'up' or 'down'")
    if not isinstance(bins,(int,list,tuple,np.ndarray)):
        raise TypeError("'bins' must be an integer or a sequence")
    if not any([coordinates=='cartesian',coordinates=='polar']):
        raise TypeError("'coordinates' must be 'cartesian' or 'polar'")
    if not isinstance(side,int):
        raise TypeError("'side' must be an integer")
    if xdomain is not None:
        if not isinstance(xdomain,(list,tuple)):
            raise TypeError("'xdomain' must be a list or tuple")
        if not len(xdomain)==2:
            raise ValueError("'xdomain' must be a two-item list or tuple")
    if ydomain is not None:
        if not isinstance(ydomain,(list,tuple)):
            raise TypeError("'xdomain' must be a list or tuple")
        if not len(ydomain)==2:
            raise ValueError("'xdomain' must be a two-item list or tuple")
    if xbins is not None:
        if not isinstance(xbins,(np.ndarray,list,tuple,int)):
            raise TypeError("'xbins' must be an integer or a sequence")
    if ybins is not None:
        if not isinstance(ybins,(np.ndarray,list,tuple,int)):
            raise TypeError("'ybins' must be an integer or a sequence")
    if not isinstance(savedir,string_types):
        raise TypeError("'savedir' must be a directory string")

    feats = [
    'brightness','saturation','hue','entropy','std','contrast',
    'dissimilarity','homogeneity','ASM','energy','correlation',
    'neural','tags'
    ]
    if feature not in feats:
        raise ValueError("""'feature' must be one of 'brightness',
        'saturation','hue','entropy','std','contrast','dissimilarity',
        'homogeneity','ASM','energy','correlation','neural', or 'tags'.""")

    if not isinstance(aggregate,bool):
        raise TypeError("'aggregate' must be True or False")
    if not isinstance(scale,bool):
        raise TypeError("'scale' must be True or False")
    if not isinstance(outline,bool):
        raise TypeError("'outline' must be True or False")
    if not isinstance(X,(pd.Series,pd.DataFrame)):
        raise TypeError("Feature matrix X must be a pandas Series or DataFrame")
    if not any([method=='kmeans',method=='hierarchical']):
        raise TypeError("'method' must be 'kmeans' or 'hierarchical'")
    if not isinstance(k,int):
        raise TypeError("'k' must be an integer")
    if not isinstance(i,int):
        raise TypeError("'i' must be an integer")
    if centroids is not None:
        if not isinstance(centroids,(list,tuple,np.ndarray)):
            raise TypeError("If passed, 'centroids' must be a sequence")

def attach(df,pathcol=None):

    if pathcol is None:
        raise ValueError("""Must supply variable 'pathcol', either as a string
            or integer that names a column of image paths in the supplied
            DataFrame, or as a pandas Series of image paths
            """)

    global ATTACHED_DATAFRAME
    global ATTACHED_PATHCOL

    ATTACHED_DATAFRAME = copy.deepcopy(df) # deep to avoid change bleed

    if isinstance(pathcol, string_types):
        ATTACHED_PATHCOL = ATTACHED_DATAFRAME[pathcol]
    elif isinstance(pathcol, int):
        try:
            ATTACHED_PATHCOL = ATTACHED_DATAFRAME[pathcol]
        except:
            ATTACHED_PATHCOL = ATTACHED_DATAFRAME.iloc[:,pathcol]
    elif isinstance(pathcol, pd.Series):
        if pathcol.index.equals(ATTACHED_DATAFRAME.index):
            ATTACHED_PATHCOL = copy.deepcopy(pathcol)
        else:
            raise ValueError("""'pathcol' must have same indices as 'df'""")
    else:
        raise TypeError("'pathcol' must be a string, integer, or pandas Series")

def detach(df):
    """Resets global variables to None. Not likely to be used often, since
    attaching a new df and pathcol just replaces the old."""

    global ATTACHED_DATAFRAME
    global ATTACHED_PATHCOL

    ATTACHED_DATAFRAME = None
    ATTACHED_PATHCOL = None

def _colfilter(pathcol,
               xcol=None,
               ycol=None,
               sample=False,
               ascending=False,
               xdomain=None,
               ydomain=None,
               facetcol=None,
               notecol=None):

    """Index matching and working with attach()"""
    pathcol = _pathfilter(pathcol)
    xcol = _featfilter(pathcol,xcol)
    ycol = _featfilter(pathcol,ycol)
    facetcol = _featfilter(pathcol,facetcol)
    notecol = _featfilter(pathcol,notecol)

    """Selecting and sorting"""
    pathcol,xcol,ycol,facetcol,notecol = _dropna(pathcol,xcol,ycol,facetcol,
                                                 notecol)
    pathcol,xcol,ycol,facetcol,notecol = _sample(pathcol,xcol,ycol,
                                                 sample,facetcol,notecol)
    pathcol,xcol,ycol,facetcol,notecol = _sort(pathcol,xcol,ycol,
                                               ascending,facetcol,notecol)
    pathcol,xcol,ycol,facetcol,notecol = _subset(pathcol,xcol,ycol,
                                                 xdomain,ydomain,facetcol,
                                                 notecol)

    return pathcol,xcol,ycol,facetcol,notecol

def _pathfilter(pathcol):

    global ATTACHED_PATHCOL

    """
    Managing user supplied data. If a DataFrame is attached, user can pass
    an empty montage function which will plot all images in attached
    DataFrame in the order they appear there. If no DataFrame is attached,
    user must supply 'pathcol', otherwise the function has no image files
    to open and plot.
    """

    if pathcol is None:
        if ATTACHED_PATHCOL is None:
            raise ValueError("No DataFrame attached; must supply 'pathcol'")
        else:
            pathcol = copy.deepcopy(ATTACHED_PATHCOL)
    else:
        if isinstance(pathcol, int): # for use in show()
            if ATTACHED_PATHCOL is None:
                raise ValueError("No DataFrame attached; must supply 'pathcol'")
            else:
                tmp = copy.deepcopy(ATTACHED_PATHCOL)
                pathcol = tmp.loc[pathcol]

    return pathcol

def _featfilter(pathcol,col):

    global ATTACHED_DATAFRAME

    if col is not None:
        if isinstance(col, pd.Series):
            if not col.index.equals(pathcol.index): # too strong a criterion?
                raise ValueError("""Image paths and image features must have
                                    same indices""")
        elif isinstance(col, (string_types, int)):
            if ATTACHED_DATAFRAME is None:
                raise TypeError("""No DataFrame attached. Feature variable
                                    must be a pandas Series""")
            else:
                tmp = copy.deepcopy(ATTACHED_DATAFRAME)
                try:
                    col = tmp[col] # if col a str or df column labels are ints
                except:
                    col = tmp.iloc[:,col] # if col is int and df col labels not
                if not col.index.equals(pathcol.index): # too strong criterion?
                    raise ValueError("""Image paths and image features must
                                        have same indices""")

    return col

def _dropna(pathcol,xcol,ycol,facetcol,notecol):
    """
    At time this function is called, all cols have same indices. We drop any
    rows with null values in each column, except notecol, which is not a
    plotting position column and so isn't strictly necessary.
    """
    setlist = []
    if xcol is not None:
        if np.mean(xcol.isnull()) > 0:
            xcol = xcol.dropna()
            setlist.append(set(xcol.index))
        else:
            setlist.append(set(xcol.index))
    if ycol is not None:
        if np.mean(ycol.isnull()) > 0:
            ycol = ycol.dropna()
            setlist.append(set(ycol.index))
        else:
            setlist.append(set(ycol.index))
    if facetcol is not None:
        if np.mean(facetcol.isnull()) > 0:
            facetcol = facetcol.dropna()
            setlist.append(set(facetcol.index))
        else:
            setlist.append(set(facetcol.index))

    if len(setlist) > 0:
        indices = set.intersection(*setlist)
        ndiff = len(pathcol.index) - len(indices)
        if ndiff > 0:
            print("removed " + str(ndiff) + " rows with missing data")
            pathcol = pathcol.loc[indices]
            if xcol is not None:
                xcol = xcol.loc[indices]
            if ycol is not None:
                ycol = ycol.loc[indices]
            if facetcol is not None:
                facetcol = facetcol.loc[indices]
            if notecol is not None:
                notecol = notecol.loc[indices]

    return pathcol,xcol,ycol,facetcol,notecol

def _sample(pathcol,xcol,ycol,sample,facetcol,notecol):

    """
    If user supplies a number for 'sample', we sample pathcol and subset
    feat/ycol to these indices. We do this because later, we might sort
    by feat/ycol. If so, we will want to then subset pathcol by feat/ycol,
    and this won't work if pathcol is missing a bunch of the indices after
    sampling. So we must sample both together.
    """
    if sample!=False:
        pathcol = pathcol.sample(n=sample)
        if xcol is not None:
            xcol = xcol.loc[pathcol.index]
        if ycol is not None:
            ycol = ycol.loc[pathcol.index]
        if facetcol is not None:
            facetcol = facetcol.loc[pathcol.index]
        if notecol is not None:
            notecol = notecol.loc[pathcol.index]

    return pathcol,xcol,ycol,facetcol,notecol

def _sort(pathcol,xcol,ycol,ascending,facetcol,notecol):

    """
    If user supplies xcol, we sort xcol and apply to path/ycol.
    At this point, if xcol is not None, we know it is a pandas Series
    (either user-supplied or from the attached df). We also know it is
    the same length as pathcol - even if pathcol got sampled above. Note
    that for scatterplots, sorting makes no difference. But it's easier
    to reuse this function for all plots. Note that your working DataFrame
    is unchanged. Also note that it is never possible to sort globally by
    ycol. This is because only histogram and scatter have ycol, and for
    histograms, this sorting happens bin by bin.
    """
    if xcol is not None:
        xcol = xcol.sort_values(ascending=ascending)
        pathcol = pathcol.loc[xcol.index]
        if ycol is not None: # nb sorting by ycol not possible globally
            ycol = ycol.loc[xcol.index]
        if facetcol is not None:
            facetcol = facetcol.loc[xcol.index]
        if notecol is not None:
            notecol = notecol.loc[xcol.index]

    return pathcol,xcol,ycol,facetcol,notecol

def _subset(pathcol,xcol,ycol,xdomain,ydomain,facetcol,notecol):
    """
    Because this function runs after pathfilter, featfilter, sample and sort,
    many checks have already taken place. If xcol and ycol are not None, we
    know they must be pandas Series and not strings.
    """

    if xdomain is not None:
        if xcol is None:
            raise ValueError("""If 'xdomain' is supplied, 'xcol' must be
                                supplied as well""")
        xcol = xcol[(xcol>=xdomain[0])&(xcol<=xdomain[1])]
        pathcol = pathcol.loc[xcol.index]
        if ycol is not None:
            ycol = ycol.loc[xcol.index]
        if facetcol is not None:
            facetcol = facetcol.loc[xcol.index]
        if notecol is not None:
            notecol = notecol.loc[xcol.index]
    if ydomain is not None:
        if ycol is None:
            raise ValueError("""If 'ydomain' is supplied, 'ycol' must be
                                supplied as well""")

        ycol = ycol[(ycol>=ydomain[0])&(ycol<=ydomain[1])]
        pathcol = pathcol.loc[ycol.index]
        xcol = xcol.loc[ycol.index] # if ycol is not None, ditto xcol
        if facetcol is not None:
            facetcol = facetcol.loc[ycol.index]
        if notecol is not None:
            notecol = notecol.loc[ycol.index]

    return pathcol,xcol,ycol,facetcol,notecol

def _bin(col,bins):
    col = pd.cut(col,bins)
    leftbinedges = [float(str(item).split(",")[0].lstrip("([")) for item in col]

    return pd.Series(leftbinedges,index=col.index)

def _facet(**kwargs):
    kwargdict = copy.deepcopy(kwargs)
    facetcol = kwargs.get('facetcol')
    pathcol = kwargs.get('pathcol')
    xcol = kwargs.get('xcol')
    ycol = kwargs.get('ycol')
    xdomain = kwargs.get('xdomain')
    ydomain = kwargs.get('ydomain')
    notecol = kwargs.get('notecol')

    """
    If we have xcol and ycol, but no user-passed domains, then we set them to
    the data extremes so that each facet can use the same domain. Conversely, if
    we do have user-passed domains, they need to dictate the plotting ranges.
    """
    if xcol is not None:
        if xdomain is None:
            xdomain = (xcol.min(),xcol.max())
    if ycol is not None:
        if ydomain is None:
            ydomain = (ycol.min(),ycol.max())

    facetlist = []
    for val in facetcol.unique():
        tmp = copy.deepcopy(kwargdict)
        tmp['facettitle'] = str(val)

        # this bit fixes plot axes across facets
        tmp['xdomain'] = xdomain
        tmp['ydomain'] = ydomain

        # generate facet
        facetedcol = facetcol[facetcol==val]
        tmp['pathcol'] = pathcol.loc[facetedcol.index]

        if xcol is not None:
            tmp['xcol'] = xcol.loc[facetedcol.index]
        if ycol is not None:
            tmp['ycol'] = ycol.loc[facetedcol.index]
        if notecol is not None:
            tmp['notecol'] = notecol.loc[facetedcol.index]

        facetlist.append(tmp)

    return facetlist
