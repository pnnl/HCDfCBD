# thanks! https://gist.github.com/rgranit/ce59711d89c64b627b4d92059d08ff82
import scipy.stats as stats
from decimal import Decimal as D

def overlap_sets(setA, setB, M, verbose=True):
    """
    Compute the probability that the sets setA and setB were generated from independent processes via a hypergeometric distribution.

    Args:
        setA (set): A set of elements
        setB (set): A set of elements
        M (int): The size of the population
        verbose (bool, optional): Whether to print the results. Defaults to True.

    Returns:
        pdensity (float): The probability density at x, the number of elements in the intersection of setA and setB
        cdf (float): The cumulative density at x.
        sf (float): The upper tail probability, the probability that the intersection of setA and setB is greater than or equal to x.
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
    
