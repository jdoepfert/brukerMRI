# pylint: disable-msg=C0103
"""This should at some point be a library with functions to import and
reconstruct Bruker MRI data.

2014, Joerg Doepfert

"""


import numpy as np

# ***********************************************************
#  class definition
# ***********************************************************
class BrukerData:
    """Class to store and process data of a Bruker MRI Experiment"""
    def __init__(self, path="", ExpNum=0, B0=9.4):
        self.method = {}
        self.acqp = {}
        self.reco = {}

        self.raw_fid = np.array([])
        self.proc_data = np.array([])
        self.k_data = np.array([])
        self.reco_data = np.array([])
        self.reco_data_norm = np.array([]) # normalized reco

        self.B0 = B0 # only needed for UFZ method
        self.GyroRatio = 0 # only needed for UFZ method
        self.ConvFreqsFactor = 0 # reference to convert Hz <--> ppm
        self.path = path
        self.ExpNum = ExpNum


    def GenerateKspace(self):
        """Reorder the data in raw_fid to a valid k-space."""

        if self.method == {}:
            raise NameError('No experiment loaded')
        elif self.method["Method"] == 'jd_UFZ_RAREst':
            self.k_data = self._GenKspace_UFZ_RARE()
        elif (self.method["Method"] == 'FLASH' or 
              self.method["Method"] == 'mic_flash'):
            self.k_data = self._GenKspace_FLASH()
        else:
            raise NameError("Unknown method")


    def ReconstructKspace(self, **kwargs):
        """Transform the kspace data to image space. If it does not yet exist,
        generate it from the raw fid. Keyword arguments [**kwargs] can
        be supplied for some methods:

        All methods:
        - KspaceCutoffIdx: list lines to be set to zero  in
          kspace prior to FT reconstruction

        jd_UFZ_RARExx:
        - NEchoes: Number of Echoes to be averaged. If NEchoes="opt",
          then the optimum number of echoes is calculated. If
          NEchoes=0, then all echoes are averaged.

        """

        # Generate k_data prior to reconstruction, if it does not yet
        # exist
        if self.k_data.size == 0:
            self.GenerateKspace()
            self._ReconstructKspace_(**kwargs)
        else:
            self._ReconstructKspace_(**kwargs)

        return self.reco_data

    def _ReconstructKspace_(self, **kwargs):
        """Select which function to use for the reco, depending on the
        method."""

        if self.method["Method"] == 'jd_UFZ_RAREst':
            self._Reco_UFZ_RARE(**kwargs)

        elif (self.method["Method"] == 'FLASH' or 
              self.method["Method"] == 'mic_flash'):
            self. _Reco_FLASH(**kwargs)

        else:
            raise NameError("Unknown method")

    # ***********************************************************
    #  method specific reordering and reco functions  start here
    # ***********************************************************   
    def _GenKspace_FLASH(self):

        complexValues = self.raw_fid
  
        NScans = (self.acqp["NI"]    # no. of images
                  * self.acqp["NAE"] # no. of experiments
                  * self.acqp["NA"]  # no. of averages
                  * self.acqp["NR"])  # no. of repetitions
    
        Matrix = self.method["PVM_Matrix"]

        kSpace = np.reshape(complexValues, (-1,Matrix[0]),
                          order="F")
        kSpace = np.reshape(kSpace, (-1, Matrix[0], Matrix[1]))
        kSpace = np.transpose(kSpace, (1,2,0))
        return kSpace

    def _Reco_FLASH(self, **kwargs):
        
        k_data = self.k_data
        reco_data = np.zeros(k_data.shape)

        for i in range(0,self.k_data.shape[2]):
            reco_data[:,:,i] = abs(fft_image(self.k_data[:,:,i]))
        
        self.reco_data = reco_data
   

    def _GenKspace_UFZ_RARE(self):

        complexValues = self.raw_fid
        complexValues = RemoveVoidEntries(complexValues,
                                          self.acqp["ACQ_size"][0])
        NEchoes = self.method["CEST_Number_Echoes"]
        NPoints = self.method["CEST_Number_SatFreqs"]
        NScans = self.method["PVM_NRepetitions"]

        return np.reshape(complexValues, (NPoints, NEchoes, NScans),
                          order="F")

    def _Reco_UFZ_RARE(self, **kwargs):

        # use pop to set default values
        KspaceCutoffIdx = kwargs.pop("KspaceCutoffIdx", [])
        NEchoes = kwargs.pop("NEchoes", "opt")

        NScans = self.method["PVM_NRepetitions"]
        NPoints = self.method["CEST_Number_SatFreqs"]
        NRecoEchoes = np.ones(NScans, dtype=np.int)

        # Determine how many echoes should be averaged
        if NEchoes == "opt": # calc opt num of echoes to be averaged

            # choose to look at real, imag, or abs part of kspace
            Data = self.k_data.real

            # find the indizes of maximum kspace signal
            MaxIndizes = []
            MaxIndizes.append(np.argmax(Data[:, 0, 0]))
            MaxIndizes.append(MaxIndizes[0] + 1
                              - 2*(Data[MaxIndizes[0]-1, 0, 0]
                                   > Data[MaxIndizes[0]+1, 0, 0]))

            # calc max of kspace echoes based on these indizes
            MaxEchoSignals = np.sum(Data[MaxIndizes, :, :], axis=0)

            # now calc opt num of echoes for each scan
            for i in range(0, NScans):
                NRecoEchoes[i] = CalcOptNEchoes(MaxEchoSignals[:, i])

            # make sure that off and on scan have same amount of
            # NRecoEchoes, i.e. echoes to be averaged
            if self.method["CEST_AcqMode"] == "On_and_Off_Scan":
                NRecoEchoes[1::2] = NRecoEchoes[0::2]

        elif NEchoes == 0: # take all echoes
            NRecoEchoes = NRecoEchoes*self.method["CEST_Number_Echoes"]

        else: # take number given by user
            NRecoEchoes = NRecoEchoes*NEchoes

        # average the echoes
        KspaceAveraged = np.zeros((NPoints, NScans), dtype=complex)
        for i in range(0, NScans):
            RecoEchoes = range(0, NRecoEchoes[i])
            KspaceAveraged[:, i] = np.mean(
                self.k_data[:, RecoEchoes, i], axis=1)
            KspaceAveraged[KspaceCutoffIdx, i] = 0

        # save reco as  FFT of the averaged kspace data
        self.reco_data, _ = FFT_center(KspaceAveraged)
        
        # normalize the data if possible
        if self.method["CEST_AcqMode"] == "On_and_Off_Scan":
            self.reco_data_norm = np.divide(abs(self.reco_data[:,1::2]), 
                                            abs(self.reco_data[:,0::2]))



# ***********************************************************
#  Functions
# ***********************************************************
def ReadExperiment(path, ExpNum):
    """Read in a Bruker MRI Experiment. Returns raw data, processed 
    data, and method and acqp parameters in a dictionary.

    """
    data = BrukerData(path, ExpNum)

    # parameter files
    data.method = ReadParamFile(path + str(ExpNum) + "/method")
    data.acqp = ReadParamFile(path + str(ExpNum) + "/acqp")
    data.reco = ReadParamFile(path + str(ExpNum) + "/pdata/1/reco")

    # processed data
    data.proc_data = ReadProcessedData(path + str(ExpNum) + "/pdata/1/2dseq",
                                       data.reco,
                                       data.acqp)

    # generate complex FID
    raw_data = ReadRawData(path + str(ExpNum) + "/fid")
    data.raw_fid = raw_data[0::2] + 1j * raw_data[1::2]

    # calculate GyroRatio and ConvFreqsFactor
    data.GyroRatio = data.acqp["SFO1"]*2*np.pi/data.B0*10**6 # in rad/Ts
    data.ConvFreqsFactor = 1/(data.GyroRatio*data.B0/10**6/2/np.pi)

    data.path = path
    data.ExpNum =ExpNum

    return data


def CalcOptNEchoes(s):
    """Find out how many echoes in an echo train [s] have to be
    included into an averaging operation, such that the signal to 
    noise (SNR) of the resulting averaged signal is maximized. 
    Based on the formula shown in the supporting information of
    the [Doepfert et al. ChemPhysChem, 15(2), 261-264, 2014]
    """

    # init vars
    s_sum = np.zeros(len(s))
    s_sum[0] = s[0]

    TestFn = np.zeros(len(s)) 
    SNR_averaged = np.zeros(len(s)) # not needed for calculation
    count = 1

    for n in range(2, len(s)+1):
        SNR_averaged = np.sum(s[0:n] / np.sqrt(n))
        s_sum[n-1] = s[n-1] + s_sum[n-2]
        TestFn[n-1] = s_sum[n-2]*(np.sqrt(float(n)/(float(n)-1))-1)
        if s[n-1] < TestFn[n-1]:
            break
        count += 1

    return count

def FFT_center(Kspace, sampling_rate=1, ax=0):
    """Calculate FFT of a time domain signal and shift the spectrum
    so that the center frequency is in the center. Additionally
    return the frequency axis, provided the right sampling frequency
    is given.
    If the data is 2D, then the FFT is performed succesively along an
    axis [ax].
    """
    FT = np.fft.fft(Kspace, axis=ax)
    spectrum = np.fft.fftshift(FT, axes=ax)
    n = FT.shape[ax]
    freq_axis = np.fft.fftshift(
        np.fft.fftfreq(n, 1/float(sampling_rate)))

    return spectrum, freq_axis

def fft_image(Kspace):
    
    return np.fft.fftshift(np.fft.fft2(Kspace))


def RemoveVoidEntries(datavector, acqsize0):
    blocksize = int(np.ceil(float(acqsize0)/2/128)*128)

    DelIdx = []
    for i in range(0, len(datavector)/blocksize):
        DelIdx.append(range(i * blocksize
                            + acqsize0/2,
                            (i + 1) * blocksize))
    return  np.delete(datavector, DelIdx)

def ReadRawData(filepath):
    with open(filepath, "r") as f:
        return np.fromfile(f, dtype=np.int32)


def ReadProcessedData(filepath, reco, acqp):
    with open(filepath, "r") as f:
        data = np.fromfile(f, dtype=np.int16)
        
        data = data.reshape(reco["RECO_size"][0],
                             reco["RECO_size"][1], -1, order="F")
        if data.ndim == 3:
            data_length = data.shape[2]
        else:
            data_length = 1

        data_reshaped = np.zeros([data.shape[1], data.shape[0], data_length])
        for i in range(0, data_length):
            data_reshaped[:, :, i] = np.rot90(data[:, :, i])

        return data_reshaped



def ReadParamFile(filepath):
    """
    Read a Bruker MRI experiment's method or acqp file to a
    dictionary.
    """
    param_dict = {}

    with open(filepath, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break

            # when line contains parameter
            if line.startswith('##$'):

                (param_name, current_line) = line[3:].split('=') # split at "="

                # if current entry (current_line) is arraysize
                if current_line[0:2] == "( " and current_line[-3:-1] == " )":
                    value = ParseArray(f, current_line)

                # if current entry (current_line) is struct/list
                elif current_line[0] == "(" and current_line[-3:-1] != " )":

                    # if neccessary read in multiple lines
                    while current_line[-2] != ")":
                        current_line = current_line[0:-1] + f.readline()

                    # parse the values to a list
                    value = [ParseSingleValue(x)
                             for x in current_line[1:-2].split(', ')]

                # otherwise current entry must be single string or number
                else:
                    value = ParseSingleValue(current_line)

                # save parsed value to dict
                param_dict[param_name] = value

    return param_dict


def ParseArray(current_file, line):

    # extract the arraysize and convert it to numpy
    line = line[1:-2].replace(" ", "").split(",")
    arraysize = np.array([int(x) for x in line])

    # then extract the next line
    vallist = current_file.readline().split()

    # if the line was a string, then return it directly
    try:
        float(vallist[0])
    except ValueError:
        return " ".join(vallist)

    # include potentially multiple lines
    while len(vallist) != np.prod(arraysize):
        vallist = vallist + current_file.readline().split()

    # try converting to int, if error, then to float
    try:
        vallist = [int(x) for x in vallist]
    except ValueError:
        vallist = [float(x) for x in vallist]

    # convert to numpy array
    if len(vallist) > 1:
        return np.reshape(np.array(vallist), arraysize)
    # or to plain number
    else:
        return vallist[0]

def ParseSingleValue(val):

    try: # check if int
        result = int(val)
    except ValueError:
        try: # then check if float
            result = float(val)
        except ValueError:
            # if not, should  be string. Remove  newline character.
            result = val.rstrip('\n')

    return result


# ***********************************************************
# -----------------------------------------------------------
# ***********************************************************


if __name__ == '__main__':

    pass

    


