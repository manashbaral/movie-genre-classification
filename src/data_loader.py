''' read .txt files

extract:
   description(x)
   labels(y)
'''

def load_train_data(file_path):
    descriptions=[]
    labels=[]

    with open(file_path,"r",encoding="utf-8") as f:
        for line in f:
            parts=line.strip().split(" ::: ")

            if len(parts)==4:
                _,title,genre,description =parts

                descriptions.append(description)
                labels.append(genre)

    return descriptions,labels

def load_test_data(file_path):
    descriptions=[]

    with open(file_path,"r",encoding="utf-8") as f:
        for line in f:
            parts=line.strip().split(" ::: ")

            if len(parts)==3:
                _,title,description =parts

                descriptions.append(description)

    return descriptions

def load_test_labels(file_path):
    labels=[]

    with open(file_path,"r",encoding="utf-8") as f:
        for line in f:
            parts=line.strip().split(" ::: ")

            if len(parts)==4:
                _,title,genre,_ =parts

                labels.append(genre)

    return labels