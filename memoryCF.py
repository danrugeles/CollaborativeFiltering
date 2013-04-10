from numpy import shape,sum,mean,array,eye,zeros
from signals import *

RESULTS=False

def getNormalizedUserWeights(ratings,SIM):
	numUsers=shape(ratings)[0]
	similarity=eye(numUsers)
	for i in range(numUsers):
		for j in range(numUsers):
			if i!=j:
				rankings_i,rankings_j=ratings[[i,j]][:,(ratings[i]!=0)*(ratings[j]!=0)]
				if SIM=="NCC":
					similarity[i][j]=ncc(rankings_i,rankings_j)
					# move weights to a 0-1 range
					similarity=(similarity+1)/2
				else:
					similarity[i][j]=cosineSimilarity(rankings_i,rankings_j)
				
	#Normalize weights
	return similarity/sum(absolute(similarity),axis=0)
			


def trainMCF(ratings,SIM):
	numUsers,numMovies=shape(ratings)
	numNonRatedMovies=sum(ratings==0,axis=1)
	usermean=mean(ratings,axis=1)
	userrankedmean=array([usermean*numMovies/(numMovies-numNonRatedMovies)]).T
	normalizedRankings=(ratings-userrankedmean)*(ratings!=0)
	normalUserWeights=getNormalizedUserWeights(ratings,SIM)

	if RESULTS:
		allPredictions=userrankedmean+dot(normalUserWeights,normalizedRankings)
		allRatings=allPredictions*(ratings==0)+ratings
		print "ratings\n",ratings
		print "newRatings\n",allRatings

	return userrankedmean,normalizedRankings


def testMFC(testSet,ratings,userrankedmean,normalizedRankings,SIM):
	numUsers,numMovies=shape(ratings)
	numTestUsers=max(testSet[:][:,0])-min(testSet[:][:,0])+1
	numMovies=1000
	#Create the user Testing matrix
	ratingsTest=zeros((numTestUsers,numMovies))

	for row in testSet:
		ratingsTest[row[0]-testSet[0][0]][row[1]-1]=row[2]

	allTestingRatings=zeros((numTestUsers,numMovies))
	#Compute for each individual block!!!!!
	for j,userdata in enumerate(ratingsTest):
		#Compute the rankedmean
		rankedmean=mean(userdata[userdata!=0])
		#Update the userrankedmeans (URM)
		URM=vstack((userrankedmean,array([[rankedmean]])))	
		#Update the normalized ratings (R)
		NR=vstack((normalizedRankings,array([userdata-rankedmean])*(userdata!=0)))
		#updateNormalizedUserweight NUW
		NUW=zeros((numUsers+1,1))
		for i in range(numUsers):
			rankings_i=ratings[[i]][:,(ratings[i]!=0)*(userdata!=0)]
			filtereduserdata=userdata[(ratings[i]!=0)*(userdata!=0)]
			if SIM=="NCC":
				NUW[i]=ncc(rankings_i,filtereduserdata)
				# move weights to a 0-1 range
				NUW=(NUW+1)/2
			else:
				NUW[i]=cosineSimilarity(rankings_i,filtereduserdata)
			
		#Normalize weights
		NUW=NUW/sum(absolute(NUW),axis=0)

		userpredictions=rankedmean+dot(NUW.T,NR)
		userratings=userpredictions*(userdata==0)+userdata
		
		allTestingRatings[j]=userratings

	return allTestingRatings


