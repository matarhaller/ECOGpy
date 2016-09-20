from math import floor, log10
import matplotlib.pyplot as plt

# Don't use LaTeX as text renderer to get text in true LaTeX
# If the two following lines are left out, Mathtext will be used
params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}

plt.rcParams.update(params)

# Define function for string formatting of scientific notation
#from http://stackoverflow.com/questions/18311909/how-do-i-annotate-with-power-of-ten-formatting
def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if not exponent:
        exponent = int(floor(log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if not precision:
        precision = decimal_digits

    #return r"${0:.{2}f}\cdot10^{{{1:d}}}$".format(coeff, exponent, precision)
    return r"${0:.{2}f}\times10^{{{1:d}}}$".format(coeff, exponent, precision)
    #return r"${0:.{2}f}e^{{{1:d}}}$".format(coeff, exponent, precision)
    #return r'$%.2f x 10^%d$' %(coeff, exponent)