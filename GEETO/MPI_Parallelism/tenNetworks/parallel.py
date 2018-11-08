import mpi4py
#from mpi4py import MPI
#from mpi4py.MPI import ANY_SOURCE
mpi4py.rc(initialize=False, finalize=False)


'''
Class for parallel functions:
	Initializes the MPI features
		Non-Blocking Exchange
		Broadcast
'''
class parallelDistributed():

	def __init__(self, MPI, param=None):
		self.mpi = MPI
		self.comm=MPI.COMM_WORLD
		self.size = self.comm.Get_size()
		# assert comm.size > 1
		self.rank = self.comm.Get_rank()
		self.name = MPI.Get_processor_name()



	'''
	Non-Blocking Exchange
		Send the params to the next network
		Receive params of the previous network
		(size + rank - 1)%size //previous // recv from prev
		(size + rank + 1)%size //next // send to next
		Returns the data of the previous network
	'''
	def nonBlockingExchange(self,data):
		reqSend1 = self.comm.isend(data, dest=((self.size+self.rank+1)%self.size), tag=self.rank)
		reqRecv2 = self.comm.irecv(source=((self.size+self.rank-1)%self.size), tag=self.rank-1)
		dataPrev = reqRecv2.wait()
		reqSend1.wait()
		print 'rank, data, dataPrev'
		print self.rank, data, dataPrev
		return dataPrev


	'''
	Get the details of the network with the minimum fitness
	the output has two values - 
		the min fitness value
		the rank of the processor having min fitness
	'''
	def getMin(self, msg):
		minloc = self.comm.allreduce(sendobj=(msg, self.rank), op=self.mpi.MINLOC)
		# minloc[0] will have the data
		# minloc[1] will have the rank
		print minloc[1], 'checking', minloc[0]
		return minloc



	'''
	Get the details of the network with the minimum fitness
	the output has two values - 
		the min fitness value
		the rank of the processor having min fitness
	'''
	def getMax(self, msg):
		maxloc = self.comm.allreduce(sendobj=(msg, self.rank), op=self.mpi.MAXLOC)
		# maxloc[0] will have the data
		# maxloc[1] will have the rank
		print maxloc[1], 'checking', maxloc[0]
		return maxloc


	'''
	Get the rank
	'''
	def getCommRank(self):
		return self.comm.Get_rank()


	'''
	Broadcast:
		Data Broadcasted from the root to all the other sockets/nodes
	'''
	def broadcast(self, data, theRank):
		data = self.comm.bcast(data, root = theRanm)
		return data

		
	'''
	Get all the data on the root
	def collateData(self, fitData, allFitness):
		print('collateData_1')
		print(fitData)
		self.comm.Allgather(fitData, allFitness)
		print('collateData_2')
		print(fitData)
		#print('collateData_3')
		#print(allFitness)
		return allFitness
	'''



'''
inp = numpy.random.rand(size)
print(inp)
senddata2 = inp[rank]
#recvdata=comm.reduce(senddata,None,root=0,op=MPI.MINLOC)
#recvdata=comm.allreduce(senddata,None,op=MPI.MIN)
senddata3 = (rank+10, rank)
senddata = rank+1
recvdata = comm.allreduce(senddata,op=MPI.PROD)
mini = comm.allreduce(senddata,op=MPI.MIN)
#minloc = comm.reduce(senddata3,None, root=0, op=MPI.MINLOC)

'''
