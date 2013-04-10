from numpy import *
import sys
WARNING=False

#-- Decrease to size ---------------------------------
#
#  Decrease the size of an array by "discretization"
#
#-------------------------------------------------
def decreaseToSize(b,size):

	newidxs=rint(arange(0,len(b),len(b)/float(size))).astype(int)

	if len(newidxs)>size:
		if WARNING: print "warning: size went up to"+str(len(newidxs))+"and it should have been"+str(size)
		return  b[newidxs[:size]]
	else:
 		return  b[newidxs]




#-- NCC ---------------------------------
#
# Between two signals of the DIFFERENT length
# The longer length is adjusted to the smaller length
#
#-------------------------------------------------
def ncc2 (a,b):
	if(len (a)== len(b)):
		return ncc(a,b)
	elif len(a)>len(b):
		newa=decreaseToSize(a,len(b))			
		return ncc(newa,b)
	else:
		newb=decreaseToSize(b,len(a))	
		return ncc(newb,a)

#-- NCC ---------------------------------
#
# Computer Normalized-Cross-Correlation
# Between two signals of the same length
#
#-------------------------------------------------
def ncc(a,b):
	a=a.astype(float)
	b=b.astype(float)
	num=((a-a.mean())*(b-b.mean())).sum()
	den=(((a-a.mean())**2).sum()*((b-b.mean())**2).sum())**0.5	
	# Warnings appear if a=[] or b=[] : invalid value encountered in double_scalars
	# In case that all ratings have the same valuefor a particular user	 
	if den==0:
		den=1	
	return num/den	

#-- CosineSimilarity ---------------------------------
#
#  Computer Vector Similarity
#
#-------------------------------------------------
def cosineSimilarity(a,b):
	a=a.astype(float)
	b=b.astype(float)
	num=dot(a,b)
	den=((a**2).sum()**0.5)*((b**2).sum()**0.5)
	# In case that all ratings have the same valuefor a particular user	 
	if den==0:
		den=1	
	
	return num/den	



if __name__=="__main__":
	a=array([2,4,2,4,2])
	b=array([1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2])
	c=array([1,2,3,3,3])
	
	print ncc2(a,-a)
	print ncc2(b,b)
	print ncc2(b,a)
	print ncc2(a,b)
	print ncc(a,a)
	print ncc(a,c)




