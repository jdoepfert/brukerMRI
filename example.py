import BrukerMRI as bruker
import matplotlib.pyplot as plt

# Display defaults
plt.rc('image', cmap='viridis', interpolation='nearest') # Display all images equally
plt.rcParams['figure.figsize'] = (16, 5)  # Widescreen figures

MainDir = "example_dataset/"
ExpNum = 100

# load FLASH experiment and reconstruct raw data
Experiment = bruker.ReadExperiment(MainDir, ExpNum)
Experiment.ReconstructKspace()

# show the resulting images
plt.figure()
for i in range(3):
	plt.subplot(1,3,i+1)
	plt.imshow(abs(Experiment.reco_data[:,:,i]))

plt.suptitle("MRI method: " + Experiment.method["Method"] + \
             ", Matrix: " + str(Experiment.method["PVM_Matrix"][0]) + \
             " x " + str(Experiment.method["PVM_Matrix"][1]))

plt.show()
