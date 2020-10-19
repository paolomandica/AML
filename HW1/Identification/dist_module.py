import numpy as np
import math



# Compute the intersection distance between histograms x and y
# Return 1 - hist_intersection, so smaller values correspond to more similar histograms
# Check that the distance range in [0,1]

def dist_intersect(x,y):
    
    # L'HO VISTO ALLA FINE, IL CHECK DELLE ASSERTION PUO' ESSERE FATTO IN QUESTO MODO O SEMPLICEMENTE
    # COME FA GALASSO NELL'ULTIMA RIGA DI QUESTO FILE, SCEGLIETE VOI
    # assert len(x)==len(y) 
    if not len(x)==len(y):       
      raise AssertionError()

    lung = len(x)
    dist = .5 * ( ([sum(x[i], y[i]) for i in range(lung)] / sum(x)) + ([sum(x[i], y[i]) for i in range(lung)] / sum(y)) )

    if not 0 <= dist and dist <= 1:
      raise AssertionError()

    return 1-dist 

# Compute the L2 distance between x and y histograms
# Check that the distance range in [0,sqrt(2)]

def dist_l2(x,y):
    if not len(x)==len(y):
        raise AssertionError()
      
    dist = sum([(x[i] - y[i])**2 for i in range(len(x))])
  
    if not 0 <= dist and dist <= np.sqrt(2):
        raise AssertionError()
  
    return dist

# Compute chi2 distance between x and y
# Check that the distance range in [0,Inf]
# Add a minimum score to each cell of the histograms (e.g. 1) to avoid division by 0

def dist_chi2(x,y):
    
    if not len(x)==len(y):
      raise AssertionError()
    
    dist = sum([((x[i] - y[i])**2) / (x[i] + y[i] + 1) for i in range(len(x))])
    
    if not 0 <= dist and dist <= math.inf:
        raise AssertionError()

    return dist


def get_dist_by_name(x, y, dist_name):
  if dist_name == 'chi2':
    return dist_chi2(x,y)
  elif dist_name == 'intersect':
    return dist_intersect(x,y)
  elif dist_name == 'l2':
    return dist_l2(x,y)
  else:
    assert False, 'unknown distance: %s'%dist_name
  




