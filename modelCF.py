from numpy import array,zeros,shape,loadtxt,arange,mean,histogram,ones,dot,random,around,sum,reshape,savetxt,loadtxt
import numpy as np
from math import *
from gaussianPDF2 import *

import sys
testingFiles=["test10.txt","test20.txt"]

class MBCF:

	def __init__(self,ratings):
		(self.numUsers, self.numMovies) = shape(ratings)
		self.ratings=ratings
		self.origRatings=ratings.copy()
		meanUser = mean(self.ratings,1)
		numRatingUser=np.sum(self.ratings!=0,axis=1)
		meanUser = np.sum(self.ratings,axis=1) / numRatingUser
		self.meanUser=meanUser.reshape((self.numUsers,1))

		normalizingratings=((self.ratings-meanUser.reshape(self.numUsers,1))**2)*array(self.ratings!=0)
		VarUser=np.sum(normalizingratings,axis=1)/np.sum(self.ratings!=0,axis=1)
		stdUser = (VarUser)**0.5
		
		self.stdUser=stdUser.reshape((self.numUsers,1))


	def normalizeRatings(self):
		self.ratings=(self.ratings-self.meanUser)/self.stdUser

	def denormalizeRatings(self,r):
		return around((r*self.stdUser)+self.meanUser)

		
	def train(self):
		#for numLatentClass in array([2, 5, 10, 20, 40, 60, 80, 100]):
		for numIterLC,numLatentClass in enumerate(array([2])):

			self.normalizeRatings()	
			
			#--------------------------------------#
			#    Initial seed for EM Algorithm
			#--------------------------------------#
			Q=random.rand(self.numUsers,self.numMovies,numLatentClass)
			Q=Q/np.sum(Q,axis=2).reshape((self.numUsers,self.numMovies,1))
			A= random.rand(self.numUsers,numLatentClass)
			B= np.sum(A,axis=1)		
			B=B.reshape((shape(B)[0],1))	
			C=ones((1,numLatentClass))
			D=dot(B,C)
			Pzu = A / D;
			Ms= random.rand(self.numMovies,numLatentClass)*2-1
			Ss= random.rand(self.numMovies,numLatentClass)*3+1
			M_yz = Ms*2-1
			Std_yz = 3*Ss+1	

			for i in range(numIteration):
				#-----------------------------------------------#
				#  Expectation step
				#----------------------------------------------#
				#--------------------------------------#
				#  1.Calculate Q
				#--------------------------------------#
				PreviousQ=Q.copy()
				for countUser in range(self.numUsers):
					for countItem in range(self.numMovies):
						down=0
						for countLC in range(numLatentClass):
							up=Pzu[countUser][countLC]*gaussianPDF2(self.ratings[countUser][countItem],M_yz[countItem][countLC],Std_yz[countItem][countLC]);
							down=down+up
							if up==0:
								print "Q is 0"
							Q[countUser][countItem][countLC]=up
						if down!=0:
							Q[countUser][countItem][:] = Q[countUser][countItem][:]/down
				#Check That Q has been updated correctly
				#print "Q"
				#print Q[:10][:,:10][:,:,0]

				#-----------------------------------------------#
				#  Mazimization step
				#----------------------------------------------#
				#--------------------------------------#
				#  1.Calculate M_yz
				#--------------------------------------#
				PreviousM = M_yz.copy();
				for countItem in range(self.numMovies):
					for countLC in range(numLatentClass):
						down=0
						up=0
						for countUser in range(self.numUsers):
							if (self.origRatings[countUser][countItem]!=0):
								up+=self.ratings[countUser][countItem]*Q[countUser][countItem][countLC]
								down+=Q[countUser][countItem][countLC]
						#Normalize for all users	
						if down!=0:
							M_yz[countItem][countLC]=up/down
						else:
							M_yz[countItem][countLC]=0
				#print "M"
				#print M_yz[:5][:5]

				#--------------------------------------#
				#  2.Calculate Std_yz
				#--------------------------------------#
				PreviousStd=Std_yz.copy()
				for countItem in range(self.numMovies):
					for countLC in range(numLatentClass):
						down=0
						tempup=0
						for countUser in range(self.numUsers):
							if (self.origRatings[countUser][countItem]!=0):
								tempup+=(self.ratings[countUser][countItem]-PreviousM[countItem][countLC])**2*Q[countUser][countItem][countLC]
								down+=Q[countUser][countItem][countLC]

						if down==0:
							Std_yz[countItem][countLC]=1
			
						elif tempup/down>0.1:
							Std_yz[countItem][countLC]=(tempup/down)**0.5
						#Standard Saturation				
						else:
							Std_yz[countItem][countLC]=0.5
				#print "Std_yz"
				#print Std_yz[:5][:5]

				#--------------------------------------#
				#  3.Calculate P(z|u) z-latent; u-user
				#--------------------------------------#
				PreviousPzu=Pzu.copy()
				for countUser in range(self.numUsers):
					down=0		
					for countLC in range(numLatentClass):
						up=0	
						for countItem in range(self.numMovies):
							if (self.origRatings[countUser][countItem]!=0):
								up+=Q[countUser][countItem][countLC]
								down+=Q[countUser][countItem][countLC]
						Pzu[countUser][countLC] = up
					Pzu[countUser][:] = Pzu[countUser][:]/down;
				#print "Pzu"
				#print Pzu[:5][:5]
		
			#--------------------------------------#
			#  TESTING
			#--------------------------------------#
			#ExpectedRating=dot(Pzu,M_yz.T)
			#ExpectedRating=self.denormalizeRatings(ExpectedRating)
			#print "Expected Rating\n",ExpectedRating[:5][:5]

		return M_yz,Std_yz


	def test(self,testSet,M_yz,Std_yz):
		numMovies,numLatentClass=shape(M_yz)
		numTestUsers=max(testSet[:][:,0])-min(testSet[:][:,0])+1

		#Create the user Testing matrix
		ratingsTest=zeros((numTestUsers,self.numMovies))
		for row in testSet:
			ratingsTest[row[0]-testSet[0][0]][row[1]-1]=row[2]


		allTestingRatings=zeros((numTestUsers,self.numMovies))

		#Compute for each individual block!!!!!
		for j,userdata in enumerate(ratingsTest):
					
			# Normalize data
			origuserdata=userdata.copy()	
			rankedmean= userdata[userdata!=0].mean()
			rankedstd=userdata[userdata!=0].std()
			if rankedstd==0:
				rankedstd=0.001
			userdata[userdata!=0]=(userdata[userdata!=0]-rankedmean)/rankedstd
				
			#--------------------------------------#
			#    Initial seed for EM Algorithm
			#--------------------------------------#
			Q = random.rand(self.numMovies, numLatentClass);
			Q=Q/np.sum(Q,axis=1).reshape((self.numMovies,1))
			A= random.rand(numLatentClass)
			Pzu = A / sum(A);
			
			for i in range(numIteration):
				#-----------------------------------------------#
				#  Expectation step
				#----------------------------------------------#
				#  Update Q
				#---------------------------------------------#
				for countItem in range(self.numMovies):
					down=0
					for countLC in range(numLatentClass):
						up=Pzu[countLC]*gaussianPDF2(userdata[countItem],M_yz[countItem][countLC],Std_yz[countItem][countLC])
						down=down+up
						if up==0:
							print "Q is 0"
						Q[countItem][countLC]=up
					if down!=0:
						Q[countItem][:] = Q[countItem][:]/down
			
				#-----------------------------------------------#
				#  Mazimization step
				#----------------------------------------------#
				#  Update P(z|u) z-latent; u-user
				#---------------------------------------------#
				Pzu=sum(Q[origuserdata!=0][:],axis=0)/sum(Q[origuserdata!=0][:])

			userpredictions=dot(array([Pzu]),M_yz.T)
			userpredictions=userpredictions*rankedstd+rankedmean
			userratings=userpredictions*(origuserdata==0)+userdata

			allTestingRatings[j]=userratings
		
			#print "Expected Rating2\n",userratings[0][10:40]
	
		return allTestingRatings

	
def createResultsFile(allTestingResults):	
	with open(testingFile,"r") as r:
		with open("result"+testingFile,"w") as w:
			for idx,query in enumerate(r.readlines()):

				query=query.strip("\n").split(" ")

				if idx==0:
					firstUserId=int(query[0])

				if query[2]=='0':
					w.write(query[0]+" "+query[1]+" "+str(int(round(allTestingResults[int(query[0])-firstUserId][int(query[1])-1])))+"\n")

	print "Results File created"


if __name__=="__main__":

	#============================
	# Custom input Parameters
	#============================

	q = 1
	numIteration = 100
	#numLatentClass = 5

	trainingFile="train.txt"
	ratings=loadtxt(trainingFile)
	TRAIN=True
	TEST=True

	#============================
	# Instantiate Model Based Filter
	#============================
	mbcf=MBCF(ratings)

	if TRAIN:
		print "Training"
		M_yz,Std_yz=mbcf.train()
		savetxt("M_yz.mat",M_yz)
		savetxt("Std_yz.mat",Std_yz)

	if TEST:
		print "Testing"
		M_yz=loadtxt("M_yz.mat")
		Std_yz=loadtxt("Std_yz.mat")
		for testingFile in testingFiles:
			testSet=loadtxt(testingFile)
			allTestingResults=mbcf.test(testSet,M_yz,Std_yz)
			createResultsFile(allTestingResults)
	

	
