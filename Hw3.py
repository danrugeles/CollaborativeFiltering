from numpy import *
from memoryCF import *
import sys


TOY=False
trainingFile="train.txt"
testingFiles=["test5.txt","test10.txt","test20.txt"]
#Cosine or NCC. Uses Cosine by default
SIM="NCC"

def createResultsFile(allTestingResults):	
	with open(testingFile,"r") as r:
		with open("result"+testingFile,"w") as w:
			for idx,query in enumerate(r.readlines()):

				query=query.strip("\n").split(" ")

				if idx==0:
					firstUserId=int(query[0])

				if query[2]=='0':
					#try:
					w.write(query[0]+" "+query[1]+" "+str(int(round(allTestingResults[int(query[0])-firstUserId][int(query[1])-1])))+"\n")
					#except:
						#print int(query[0])-firstUserId,int(query[1])-1
						#sys.exit()
	print "File created"

	
if __name__=="__main__":

	#**TRAINING
	if TOY:
		ratings=array([[1,2,1,2,3],[5,5,0,4,4],[1,0,5,0,0]])
	else:
		ratings=loadtxt(trainingFile)

	userrankedmean,normalizedRankings=trainMCF(ratings,SIM)

	#**TESTING
	if not TOY:
		for testingFile in testingFiles:
			testSet=loadtxt(testingFile)
			allTestingResults=testMFC(testSet,ratings,userrankedmean,normalizedRankings,SIM)
			createResultsFile(allTestingResults)
