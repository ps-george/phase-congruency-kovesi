import numpy as np
import cv2

# References:
#
#     Peter Kovesi, "Imag5 Features From Phase Congruency". Videre: A
#     Journal of Computer Vision Research. MIT Press. Volume 1, Number 3,
#     Summer 1999 http://mitpress.mit.edu/e-journals/Videre/001/v13.html
#
#     Peter Kovesi, "Phase Congruency Detects Corners and
#     Edges". Proceedings DICTA 2003, Sydney Dec 10-12

# Copyright (c) 1996-2017 Peter Kovesi
# Centre for Exploration Targeting
# The University of Western Australia
# peter.kovesi at uwa edu au
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# The Software is provided "as is", without warranty of any kind.


# based on phasecong3.m by Peter Kovesi
def phasecong3(input_img, nscales=3, norient=6, minWaveLength=3, mult=2.1, sigmaOnF=0.55, k=2.0, cutOff=0.5, g=10, noiseMethod=-1):

    # nscales          4    - Number of wavelet scales, try values 3-6.
    #                         should be odd (choose 3 to match wavelet features used for correlation experiments)
    # norient          6    - Number of filter orientations.
    # minWaveLength    3    - Wavelength of smallest scale filter.
    # mult             2.1  - Scaling factor between successive filters.
    # sigmaOnF         0.55 - Ratio of the standard deviation of the Gaussian
    #                         describing the log Gabor filter's transfer function
    #                         in the frequency domain to the filter center frequency.
    # k                2.0  - No of standard deviations of the noise energy beyond
    #                         the mean at which we set the noise threshold point.
    #                         You may want to vary this up to a value of 10 or
    #                         20 for noisy images
    # cutOff           0.5  - The fractional measure of frequency spread
    #                         below which phase congruency values get penalized.
    # g                10   - Controls the sharpness of the transition in
    #                         the sigmoid function used to weight phase
    #                         congruency for frequency spread.
    # noiseMethod      -1   - Parameter specifies method used to determine
    #                         noise statistics.
    #                           -1 use median of smallest scale filter responses
    #                           -2 use mode of smallest scale filter responses
    #                           0+ use noiseMethod value as the fixed noise threshold
    #                              to save recomputing it.

    epsilon = .0001 # Used to prevent division by zero.

    # [rows,cols] = size(im);
    # imagefft = fft2(im);              % Fourier transform of image
    #
    # zero = zeros(rows,cols);
    # EO = cell(nscale, norient);       % Array of convolution results.
    # PC = cell(norient,1);
    # covx2 = zero;                     % Matrices for covariance data
    # covy2 = zero;
    # covxy = zero;
    #
    # EnergyV = zeros(rows,cols,3);     % Matrix for accumulating total energy
    #                                   % vector, used for feature orientation
    #                                   % and type calculation
    #
    # pcSum = zeros(rows,cols);

    f_cv = cv2.dft(np.float32(input_img),flags=cv2.DFT_COMPLEX_OUTPUT)

    #------------------------------
    # Initialise variables
    nrows, ncols = input_img.shape
    zero = np.zeros((nrows,ncols))
    EO = np.zeros((nrows,ncols,nscales,norient),dtype=complex)
    PC = np.zeros((nrows,ncols,norient))
    covx2 = np.zeros((nrows,ncols))
    covy2 = np.zeros((nrows,ncols))
    covxy = np.zeros((nrows,ncols))
    EnergyV = np.zeros((nrows,ncols,3))
    pcSum = np.zeros((nrows,ncols))

    # Matrix of radii
    cy = int(np.floor(nrows/2))
    cx = int(np.floor(ncols/2))
    y, x = np.mgrid[0:nrows, 0:ncols]
    y = (y-cy)/nrows
    x = (x-cx)/ncols

    radius = np.sqrt(x**2 + y**2)
    radius[cy, cx] = 1 # set center of radius to 1

    # Matrix values contain polar angle.
    # (note -ve y is used to give +ve anti-clockwise angles)
    theta = np.arctan2(-y, x)
    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    # Initialise set of annular bandpass filters
    annular_bandpass_filters = np.empty((nrows,ncols,nscales))

    #p = np.arange(nscales) - int(np.floor(nscales/2))
    #fSetCpo = CriticalBandCyclesPerObject*mult**p
    #fSetCpi = fSetCpo * ObjectsPerImage

    # Ratio of angular interval between filter orientations and the standard deviation
    # of the angular Gaussian function used to construct filters in the freq. plane.

    # dThetaOnSigma = 1.3
    # # The standard deviation of the angular Gaussian function used to construct filters in the frequency plane.
    # thetaSigma = np.pi / norient / dThetaOnSigma
    # BandpassFilters = np.empty((nrows,ncols,nscales,norient))
    # evenWavelets = np.empty((nrows,ncols,nscales,norient))
    # oddWavelets  = np.empty((nrows,ncols,nscales,norient))

    # The following implements the log-gabor transfer function
    """ From http://www.peterkovesi.com/matlabfns/PhaseCongruency/Docs/convexpl.html
        The filter bandwidth is set by specifying the ratio of the standard deviation
        of the Gaussian describing the log Gabor filter's transfer function in the
        log-frequency domain to the filter center frequency. This is set by the parameter
        sigmaOnF . The smaller sigmaOnF is the larger the bandwidth of the filter.
        I have not worked out an expression relating sigmaOnF to bandwidth, but
        empirically a sigmaOnF value of 0.75 will result in a filter with a bandwidth
        of approximately 1 octave and a value of 0.55 will result in a bandwidth of
        roughly 2 octaves.
    """
    # sigmaOnF = 0.74  # approximately 1 octave
    # sigmaOnF = 0.55  # approximately 2 octaves
    """ From Wilson, Loffler and Wilkinson (2002 Vision Research):
        The bandpass filtering alluded to above was used because of ubiquitous evidence
        that face discrimination is optimal within a 2.0 octave (at half amplitude)
        bandwidth centered upon 8–13 cycles per face width (Costen et al., 1996;
        Fiorentini et al., 1983; Gold et al., 1999; Hayes et al., 1986; Näsänen, 1999).
        We therefore chose a radially symmetric filter with a peak frequency of 10.0
        cycles per mean face width and a 2.0 octave bandwidth described by a difference
        of Gaussians (DOG)
    """
    # Lowpass filter to remove high frequency 'garbage'
    filterorder = 15  # filter 'sharpness'
    cutoff = .45
    normradius = radius / (abs(x).max()*2)
    # Note: lowpassbutterworth is currently DC centered.
    lowpassbutterworth = 1.0 / (1.0 + (normradius / cutoff)**(2*filterorder))

    for s in np.arange(nscales):
        wavelength = minWaveLength*mult**s
        fo = 1.0/wavelength                     # Centre frequency of filter.
        logGabor = np.exp((-(np.log(radius/fo))**2) / (2 * np.log(sigmaOnF)**2))
        annular_bandpass_filters[:,:,s] = logGabor*lowpassbutterworth  # Apply low-pass filter
        annular_bandpass_filters[cy,cx,s] = 0   # Set the value at the 0 frequency point of the filter
                                                # back to zero (undo the radius fudge).

    # main loop
    # filterOrient = np.arange(start=0, stop=np.pi - np.pi / norient, step = np.pi / norient)
    for o in np.arange(norient):
        # Construct the angular filter spread function
        # angl = filterOrient[o]
        angl = o*np.pi/norient # Filter angle, simpler way of calculating

        # For each point in the filter matrix calculate the angular distance from
        # the specified filter orientation. To overcome the angular wrap-around
        # problem sine difference and cosine difference values are first computed
        # and then the atan2 function is used to determine angular distance.
        ds = sintheta * np.cos(angl) - costheta * np.sin(angl)      # Difference in sine.
        dc = costheta * np.cos(angl) + sintheta * np.sin(angl)      # Difference in cosine.
        dtheta = np.abs(np.arctan2(ds,dc))                              # Absolute angular distance.

        # Scale theta so that cosine spread function has the right wavelength
        # and clamp to pi
        np.minimum(dtheta*norient/2, np.pi, out=dtheta)

        # The spread function is cos(dtheta) between -pi and pi.  We add 1,
        # and then divide by 2 so that the value ranges 0-1
        spread = (np.cos(dtheta)+1)/2

        sumE_ThisOrient   = np.zeros((nrows,ncols))  # Initialize accumulator matrices.
        sumO_ThisOrient   = np.zeros((nrows,ncols))
        sumAn_ThisOrient  = np.zeros((nrows,ncols))
        energy            = np.zeros((nrows,ncols))

        maxAn = []
        for s in np.arange(nscales):
            filter = annular_bandpass_filters[:,:,s] * spread # Multiply radial and angular
                                                              # components to get the filter.

            criticalfiltershift = np.fft.ifftshift(filter)
            criticalfiltershift_cv = np.empty((nrows, ncols, 2))
            for ip in range(2):
                criticalfiltershift_cv[:,:,ip] = criticalfiltershift

            # Convolve image with even and odd filters returning the result in EO
            MatrixEO = cv2.idft(criticalfiltershift_cv * f_cv)
            EO[:,:,s,o] = MatrixEO[:,:,1] + 1j*MatrixEO[:,:,0]

            An = cv2.magnitude(MatrixEO[:,:,0], MatrixEO[:,:,1]) # Amplitude of even & odd filter response.

            sumAn_ThisOrient = sumAn_ThisOrient + An            # Sum of amplitude responses.
            sumE_ThisOrient = sumE_ThisOrient + MatrixEO[:,:,1] # Sum of even filter convolution results.
            sumO_ThisOrient = sumO_ThisOrient + MatrixEO[:,:,0] # Sum of odd filter convolution results.

            # At the smallest scale estimate noise characteristics from the
            # distribution of the filter amplitude responses stored in sumAn.
            # tau is the Rayleigh parameter that is used to describe the
            # distribution.
            if s == 0:
                # if noiseMethod == -1:     # Use median to estimate noise statistics
                tau = np.median(sumAn_ThisOrient) / np.sqrt(np.log(4))
                # elif noiseMethod == -2: # Use mode to estimate noise statistics
                #    tau = rayleigh_mode(sumAn_ThisOrient)
                maxAn = An
            else:
                # Record maximum amplitude of components across scales.  This is needed
                # to determine the frequency spread weighting.
                np.maximum(maxAn,An,out=maxAn)
        # complete scale loop
        
        # next section within mother (orientation) loop
        # Accumulate total 3D energy vector data, this will be used to
        # determine overall feature orientation and feature phase/type
        EnergyV[:,:,0] = EnergyV[:,:,0] + sumE_ThisOrient
        EnergyV[:,:,1] = EnergyV[:,:,1] + np.cos(angl)*sumO_ThisOrient
        EnergyV[:,:,2] = EnergyV[:,:,2] + np.sin(angl)*sumO_ThisOrient

        # Get weighted mean filter response vector, this gives the weighted mean
        # phase angle.
        XEnergy = np.sqrt(sumE_ThisOrient**2 + sumO_ThisOrient**2) + epsilon
        MeanE = sumE_ThisOrient / XEnergy
        MeanO = sumO_ThisOrient / XEnergy

        # Now calculate An(cos(phase_deviation) - | sin(phase_deviation)) | by
        # using dot and cross products between the weighted mean filter response
        # vector and the individual filter response vectors at each scale.  This
        # quantity is phase congruency multiplied by An, which we call energy.

        for s in np.arange(nscales):
            # Extract even and odd convolution results.
            E = EO[:,:,s,o].real
            O = EO[:,:,s,o].imag

            energy= energy+ E*MeanE + O*MeanO - np.abs(E*MeanO - O*MeanE)

        ## Automatically determine noise threshold
        #
        # Assuming the noise is Gaussian the response of the filters to noise will
        # form Rayleigh distribution.  We use the filter responses at the smallest
        # scale as a guide to the underlying noise level because the smallest scale
        # filters spend most of their time responding to noise, and only
        # occasionally responding to features. Either the median, or the mode, of
        # the distribution of filter responses can be used as a robust statistic to
        # estimate the distribution mean and standard deviation as these are related
        # to the median or mode by fixed constants.  The response of the larger
        # scale filters to noise can then be estimated from the smallest scale
        # filter response according to their relative bandwidths.
        #
        # This code assumes that the expected reponse to noise on the phase congruency
        # calculation is simply the sum of the expected noise responses of each of
        # the filters.  This is a simplistic overestimate, however these two
        # quantities should be related by some constant that will depend on the
        # filter bank being used.  Appropriate tuning of the parameter 'k' will
        # allow you to produce the desired output.

        noise_threshold = noiseMethod   # use supplied noiseMethod value as the threshold
        if noiseMethod < 0:
            # Estimate the effect of noise on the sum of the filter responses as
            # the sum of estimated individual responses (this is a simplistic
            # overestimate). As the estimated noise response at succesive scales
            # is scaled inversely proportional to bandwidth we have a simple
            # geometric sum.
            totalTau = tau * (1 - (1/mult)**nscales)/(1-(1/mult))

            # Calculate mean and std dev from tau using fixed relationship
            # between these parameters and tau. See
            # http://mathworld.wolfram.com/RayleighDistribution.html
            EstNoiseEnergyMean = totalTau*np.sqrt(np.pi/2)        # Expected mean and std
            EstNoiseEnergySigma = totalTau*np.sqrt((4-np.pi)/2)   # values of noise energy
            noise_threshold =  EstNoiseEnergyMean + k*EstNoiseEnergySigma # Noise threshold

        # Apply noise threshold,  this is effectively wavelet denoising via
        # soft thresholding.
        np.maximum(energy - noise_threshold, 0, out=energy)

        # Form weighting that penalizes frequency distributions that are
        # particularly narrow.  Calculate fractional 'width' of the frequencies
        # present by taking the sum of the filter response amplitudes and dividing
        # by the maximum amplitude at each point on the image.   If
        # there is only one non-zero component width takes on a value of 0, if
        # all components are equal width is 1.
        width = (sumAn_ThisOrient/(maxAn + epsilon) - 1) / (nscales-1)

        # Now calculate the sigmoidal weighting function for this orientation.
        weight = 1.0 / (1 + np.exp( (cutOff - width)*g))

        # Apply weighting to energy and then calculate phase congruency
        PC[:,:,o] = weight*energy/sumAn_ThisOrient   # Phase congruency for this orientatio

        pcSum = pcSum + PC[:,:,o]

        # Build up covariance data for every point
        covx = PC[:,:,o]*np.cos(angl)
        covy = PC[:,:,o]*np.sin(angl)
        covx2 = covx2 + covx**2
        covy2 = covy2 + covy**2
        covxy = covxy + covx*covy
        # above everyting within orientaiton loop
    # ------------------------------------------------------------------------
    # current work
    # Edge and Corner calculations
    # The following is optimised code to calculate principal vector
    # of the phase congruency covariance data and to calculate
    # the minimumum and maximum moments - these correspond to
    # the singular values.

    # First normalise covariance values by the number of orientations/2
    covx2 = covx2/(norient/2)
    covy2 = covy2/(norient/2)
    covxy = 4*covxy/norient   # This gives us 2*covxy/(norient/2)
    denom = np.sqrt(covxy**2 + (covx2-covy2)**2)+epsilon
    edges_M = (covy2+covx2 + denom)/2          # Maximum moment
    corners_m = (covy2+covx2 - denom)/2          # ... and minimum moment

    # Orientation and feature phase/type computation
    # ORM = np.arctan2(EnergyV[:,:,2], EnergyV[:,:,1])
    # ORM[ORM<0] = ORM[ORM<0]+np.pi       # Wrap angles -pi..0 to 0..pi
    # ORM = np.round(ORM*180/np.pi)        # Orientation in degrees between 0 and 180

    # OddV = np.sqrt(EnergyV[:,:,1]**2 + EnergyV[:,:,2]**2)
    # featType = np.arctan2(EnergyV[:,:,0], OddV)  # Feature phase  pi/2 <-> white line,
                                            # 0 <-> step, -pi/2 <-> black line
    # ------------------------------------------------------------------------

    #return edges_M, corners_m, ORM, EO, noise_threshold, annular_bandpass_filters, lowpassbutterworth
    return edges_M, corners_m #, noise_threshold

## -------------------------------------------------------------------------
# RAYLEIGHMODE
#
# Computes mode of a vector/matrix of data that is assumed to come from a
# Rayleigh distribution.
#
# Usage:  rmode = rayleighmode(data, nbins)
#
# Arguments:  data  - data assumed to come from a Rayleigh distribution
#             nbins - Optional number of bins to use when forming histogram
#                     of the data to determine the mode.
#
# Mode is computed by forming a histogram of the data over 50 bins and then
# finding the maximum value in the histogram.  Mean and standard deviation
# can then be calculated from the mode as they are related by fixed
# constants.
#
# mean = mode * sqrt(pi/2)
# std dev = mode * sqrt((4-pi)/2)
# 
# See
# http://mathworld.wolfram.com/RayleighDistribution.html
# http://en.wikipedia.org/wiki/Rayleigh_distribution
#

# def rayleigh_mode(data, nbins=50):
#     max_data = np.max(data)
#     # edges = 0:max_data / nbins:max_data # python equivalent is 
#     # python equivalent:
#     a = np.array(range(0, max_data))
#     b = np.array(range(nbins, max_data))
#     edges = np.linalg.lstsq(b.T, a.T)[0]
#     edges = np.dot(a, np.linalg.pinv(b)) 
#     n = np.digitize(data, edges) 
#     ind  = np.argmax(n) # Find maximum and index of maximum in histogram 
#     # hist_max = n[ind]
#     rmode = (edges[ind]+edges[ind+1])/2
#     return rmode
