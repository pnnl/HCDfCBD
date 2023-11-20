# thanks! https://gist.github.com/rgranit/ce59711d89c64b627b4d92059d08ff82
import scipy.stats as stats
from decimal import Decimal as D

def overlap_sets(setA, setB, M, verbose=True):
    """
    Accepts to lists
    M is the population size (previously N)
    n is the number of successes in the population 
    N is the sample size (previously n)
    x is still the number of drawn “successes”
    """
    n= len(setA)
    N= len(setB)
    x= len(setA.intersection(setB))
 
    rv = stats.hypergeom(M, n, N)

    pdensity = rv.pmf(x)
    cdf = rv.cdf(x)
    sf = rv.sf(x-1)

    if verbose:
        print(f"p(x) : {rv.pmf(x)}")
        print('p-value <= ' + str(x) + ': ' + str(cdf))
        print('p-value >= ' + str(x)  + ': ' + str(sf))

    return pdensity, cdf, sf
    
