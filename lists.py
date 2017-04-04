import json
import sys 

from pprint import pprint
with open('drums_ALL.json') as data_file:
	data = json.load(data_file);

def gen_runup(size):
    temp = []
    for i in range(size):
#    temp = i/size
        # temp.append((i/16.0, i/16.0, i/16.0, (i % 4.0)/3.0))
        # temp.append((0, 0, 0, (i % 4.0)/3.0))
          temp.append(((i % 4.0)/3.0, (i % 4.0)/3.0, (i % 4.0)/3.0, (i % 4.0)/3.0))


    return temp
    # return []

def get_data():
	count = 0

	patterns = []

	for drum in data["data"]:
		drumPat = gen_runup(16)
        
		for i in range(len(drum["k"])):
	        #drumPat.append((int(drum["k"][i]), int(drum["s"][i]), int(drum["h"][i])))
			drumPat.append((int(drum["k"][i]), int(drum["s"][i]), int(drum["h"][i]), (i % 4.0)/3.0)) # to print index

		# just in case, append first slice to the end for 64+1
		#drumPat.append((int(drum["k"][0]), int(drum["s"][0]), int(drum["h"][0]), (i % 4.0)/3.0))
		patterns.append(drumPat)


	return patterns

