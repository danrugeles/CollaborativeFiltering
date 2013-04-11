import math as m
def gaussianPDF2(v, mean, std):
	if std != 0:
		p = 1/m.sqrt(2*m.pi)*m.exp(-0.5*((v-mean)/std)**2) /std;
	else :
		if abs(v - mean) < 0.00001 :
			p = 1;
		else:	
			p =0;
	return p


if __name__=="__main__":
	print gaussianPDF2(1,0,1)
