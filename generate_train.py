import os

#set Image Directory path 
image_path = '/home/erin/Documents/project/darknet/data/Apple_Banana_Orange'  #copy annotated data and paste to the new folder 
os.chdir(image_path)

#Iterate through each image found in the directory and save the corresponding path to the list called as path_list
path_list=[]
#Go through all the image files in the directory 
#Fullstop in os.walk('.)means the current directory
for current_dir,dirs,files in os.walk('.'):
    #iterate through all the files
    for f in files:
        #check if the file extension ends with '.jpg'
        if f.endswith('.jpg'):
            #prepare file path to save into train.txt
            file_loc=image_path + '/' + f 
            #append the path data into list "path_list". New line character \n is used to write the new content 
            path_list.append(file_loc + '\n')

#Divide the data into 80:20 ratio. Get 20% of data from path_list
# to write into the test.txt file 
path_list_test = path_list[:int(len(path_list)*0.20)]
#delete the same 20% records from the path_list as that 20% data is in path_list_test now
path_list = path_list[int(len(path_list)*0.20):]


#create train.txt file and write 80% of data(lines) inside it
with open ('train.txt', 'w') as train:
    #Iterate through all the elements in the list
    for i in path_list:
        #write the current path at the end of the file
        train.write(i)

#create text.txt file and write 20% of data (lines) inside it
with open('test.txt','w') as test:
    #Iterate through all the elements in the list 
    for i in path_list_test:
        #write the current path at the end of the file
        test.write(i)


# Initialize the counter 
i=0
#create classes.names file by reading content from existing classes.txt file 
with open(image_path + '/'+'classes.names','w') as cls, open(image_path + '/'+ 'classes.txt','r') as text:
    #iterate through indivisual lines in classes.txt file and write them into classes.name file
    for l in text:
        cls.write(l)
        #increasing counter 
        i+=1

#create image_data.data 
with open(image_path + '/' +'image_data.data','w') as data:
    #write number of classes 
    data.write('classes = '+ str(i)+'\n')
    #write fully qualified path of the train.txt file
    data.write('train='+image_path + '/'+'train.txt'+'\n')
    #write fully qualified path of the train.txt file
    data.write('valid='+image_path + '/'+'test.txt'+'\n')
    #write fully qualified path of the classes.names file 
    data.write('names='+image_path+'/'+'classes.names'+ '\n')
    #specify folder path to save trained model weights
    data.write('backup = backup')