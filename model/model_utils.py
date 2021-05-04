import torch
import pdb
import math
from torch.autograd import Variable
import torch.autograd as autograd
import matplotlib
import numpy as np


def UpdateMultiTaskWeightWithAlphas(index, modelMain):
    
    total_weight = [[],[],[]]
    total_weight[0] = modelMain[index].fisherMatrix[0] / float(index + 1)
    total_weight[1] = modelMain[index].fisherMatrix[1] / float(index + 1)
    total_weight[2] = modelMain[index].fisherMatrix[2] / float(index + 1)

    for temp_index in range(index):

        total_weight[0] += modelMain[temp_index].fisherMatrix[0] / float(index + 1)
        total_weight[1] += modelMain[temp_index].fisherMatrix[1] / float(index + 1)
        total_weight[2] += modelMain[temp_index].fisherMatrix[2] / float(index + 1)


    for temp_index in range(index + 1):
        
        modelMain[temp_index].LW[0] = modelMain[temp_index].fisherMatrix[0] / total_weight[0] / float(index + 1)
        modelMain[temp_index].LW[1] = modelMain[temp_index].fisherMatrix[1] / total_weight[1] / float(index + 1)
        modelMain[temp_index].LW[2] = modelMain[temp_index].fisherMatrix[2] / total_weight[2] / float(index + 1)
    
    
def calculateFisherMatrix(self, mb=1):
    FM = []
    x = self.cl_dataset.get_train_data(self.global_index)[0]
    data_size = x.shape[0]
    total_step = int(math.ceil(float(data_size)/mb))
    # pdb.set_trace()
    FM.append(torch.zeros(self.modelMain[self.global_index].weight1.shape).cuda())
    FM.append(torch.zeros(self.modelMain[self.global_index].weight2.shape).cuda())
    FM.append(torch.zeros(self.modelMain[self.global_index].weight3.shape).cuda())

    for step in range(total_step):
        ist = (step * mb) % data_size
        ied = min(ist + mb, data_size)
        # y_sample = tf.reshape(tf.one_hot(tf.multinomial(self.y, 1), 10), [-1, 10])
        interpolates = Variable(self.Tensor(x[ist:ied]), requires_grad=True)

        disc_interpolates, _, _ = self.modelMain[self.global_index](interpolates)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                            grad_outputs=torch.ones(
                                disc_interpolates.size()).cuda(),
                            create_graph=True, retain_graph=True, only_inputs=False)

        
        # pdb.set_trace()
        
        FM[0] += self.modelMain[self.global_index].weight1.grad ** 2
        FM[1] += self.modelMain[self.global_index].weight2.grad ** 2
        FM[2] += self.modelMain[self.global_index].weight3.grad ** 2

    
    FM[0] += 1e-8
    FM[1] += 1e-8
    FM[2] += 1e-8
    # pdb.set_trace()
    self.modelMain[self.global_index].fisherMatrix = FM


def CalculateModeParametes(self):

    for label_index in range(self.global_index + 1):
        
        self.modelMaintainMode.addmmtoself(self.modelMain[label_index].LW, self.modelMain[label_index])
    

def CalculateModePerClassParametes(self):

    for label_index in range(self.global_index + 1):
    	self.modelMain[label_index].share_model_mode.__init__()
    	for temp_label in range(self.global_index + 1):

        	if temp_label == label_index:
        		continue
        	else:
        		self.modelMain[label_index].share_model_mode.addmmtoself(self.modelMain[temp_label].LW, self.modelMain[temp_label])




def Update_Gradient(self, index):

	# pdb.set_trace()
	self.modelMain[index].weight1.grad.data *=  (self.modelMain[index].fisherMatrix[0].data.max() - self.modelMain[index].fisherMatrix[0].data)
	self.modelMain[index].weight2.grad.data *=  (self.modelMain[index].fisherMatrix[1].data.max() - self.modelMain[index].fisherMatrix[1].data)
	self.modelMain[index].weight3.grad.data *=  (self.modelMain[index].fisherMatrix[2].data.max() - self.modelMain[index].fisherMatrix[2].data)


	# pass

def FindErrors(results, labels, label_index, opt):

	values, indices = labels.sort()
	values = values.cpu().int()
	results_ = results[indices]
	width = results_.shape[0]
	length = values[-1] + 1

	outcome = torch.zeros(length, width).scatter(0, results_.cpu().long().view(1,-1), 1)

	finalMa = torch.zeros(length, length)
	for i in range(length):
		# pdb.set_trace()
		finalMa[:,i] = torch.sum(outcome[:,values==i], dim=1)

	plots(finalMa, values, label_index, opt)

def plots(img, label, label_index, opt):
	# vegetables = ["cucumber", "tomato", "lettuce", "asparagus", "potato", "wheat", "barley"]
	# farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening", "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

	harvest =img.numpy()
	
	import matplotlib.pyplot as plt

	fig, ax = plt.subplots()
	im = ax.imshow(harvest)
	size = img.shape
	# We want to show all ticks...
	# ax.set_xticks(np.arange(size[1]))
	# ax.set_yticks(np.arange(size[0]))
	# ... and label them with the respective list entries
	# ax.set_xticklabels([ str(x) for x in label.numpy().tolist()])
	# ax.set_yticklabels([ str(x) for x in label.numpy().tolist()])
	# pdb.set_trace()

	# Rotate the tick labels and set their alignment.
	# plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	# for i in range(size[0]):
	# 	for j in range(size[1]):
	# 		text = ax.text(j, i, harvest[i, j], ha="center", va="center", color="w")

	ax.set_title("Harvest of local farmers (in tons/year)")
	# fig.tight_layout()
	plt.savefig('./img/' + str(opt.gpu) + '/' + str(label_index) + '.eps', format='eps', bbox_inches='tight')

	plt.close()
















































