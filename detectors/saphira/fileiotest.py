import numpy as np

with open('test.text', "r+") as myfile:
    contents = myfile.read().split('\n')
    
for i in range(len(contents)):
    if 'filename' in contents[i]:
        rfi = contents[i+1].replace('[', '').replace(']', '') #remove brackets
        rfi = [int(numeric_string) for numeric_string in rfi.split(' ')]
        rfi = np.append(rfi, 1234)
        break
    
newcontents = np.append(contents[:i]
myfile.write(contents)
