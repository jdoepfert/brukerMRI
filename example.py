import BrukerMRI as bruker
import pylab as pl

MainDir = "example_dataset/"
ExpNum = 100

# load FLASH experiment and reconstruct raw data
Experiment = bruker.ReadExperiment(MainDir, ExpNum)
Experiment.ReconstructKspace()

# show the resulting images
pl.figure()
pl.subplot(131)
pl.imshow(abs(Experiment.reco_data[:,:,0]))

pl.subplot(132)
pl.imshow(abs(Experiment.reco_data[:,:,1]))

pl.subplot(133)
pl.imshow(abs(Experiment.reco_data[:,:,2]))

pl.suptitle("MRI method: " + Experiment.method["Method"] + \
            ", Matrix: " + str(Experiment.method["PVM_Matrix"][0]) + \
            " x " + str(Experiment.method["PVM_Matrix"][1]))

pl.show()
