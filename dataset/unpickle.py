import pickle
infile=open('flowdata.pickle','rb')
newdict=pickle.load(infile)
infile.close()

print(new_dict.decode("utf8"))
print(type(new_dict))
